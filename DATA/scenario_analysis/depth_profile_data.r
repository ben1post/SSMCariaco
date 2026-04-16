# =============================================================================
# DEPTH PROFILE DATA FUNCTIONS
# For flexible depth integration with scenario-specific cutoffs
#
# This script provides functions to:
# 1. Load pre-interpolated HPLC pigment profiles
# 2. Interpolate Niskin data on-demand (cached)
# 3. Merge with scenario classification
# 4. Fit euphotic depth proxy model (EuZ ~ Isotherm_21)
# 5. Integrate to flexible depth bounds (fixed / scenario / per-date)
# 6. Calculate size fractions from integrated pigments
#
# Usage (from Jupyter notebook in scenario_analysis/):
#   source("depth_profile_data.R")
#   
#   # Load and merge all data
#   profile_data <- load_profile_data()
#   
#   # Fit proxy model
#   proxy_model <- fit_euphotic_proxy(profile_data$scenario)
#   
#   # Integrate with different strategies
#   integrated_fixed <- integrate_to_depth(profile_data, depth_mode = "fixed", fixed_depth = 50)
#   integrated_scenario <- integrate_to_depth(profile_data, depth_mode = "scenario")
#   integrated_dynamic <- integrate_to_depth(profile_data, depth_mode = "dynamic", proxy_model = proxy_model)
# =============================================================================

library(tidyverse)
library(oce)

# Source dependencies
source("../interpolateData.r")
source("scenario_classification.R")

# =============================================================================
# 1. DATA LOADING FUNCTIONS
# =============================================================================

#' Load pre-interpolated HPLC depth profiles
#'
#' @return Data frame with columns: date, depth, and all pigment variables
load_hplc_profiles <- function() {

  hplc_profiles <- readRDS("../processed/HPLC_depth_profiles.rds")
  
  cat(sprintf("Loaded HPLC profiles: %d dates, depths 0-%dm\n", 
              length(unique(hplc_profiles$date)), 
              max(hplc_profiles$depth)))
  

  return(hplc_profiles)
}

#' Load and interpolate Niskin data
#'
#' Interpolates key variables to 0-200m at 1m resolution.
#' Results are cached to avoid re-computation.
#'
#' @param force_recompute If TRUE, recompute even if cache exists
#' @return Data frame with columns: date, depth, and interpolated variables
load_niskin_profiles <- function(force_recompute = FALSE) {

  cache_file <- "../processed/Niskin_depth_profiles.rds"
  

  # Use cached version if available

  if (file.exists(cache_file) && !force_recompute) {
    niskin_profiles <- readRDS(cache_file)
    cat(sprintf("Loaded cached Niskin profiles: %d dates, depths 0-%dm\n",
                length(unique(niskin_profiles$date)),
                max(niskin_profiles$depth)))
    return(niskin_profiles)
  }
  
  # Otherwise, interpolate from cleaned data
  cat("Interpolating Niskin data (this may take a moment)...\n")
  
  niskin_ds <- readRDS("../processed/Niskin_cleaned.rds")
  
  niskin_vars <- c("NO3_merged", "Chlorophyll", "PrimaryProductivity", 
                   "PN_ug_L", "Temperature")
  
  niskin_interpolated_list <- list()
  
  for (variable in niskin_vars) {
    cat(sprintf("  Interpolating %s...\n", variable))
    temp_int <- interpolateDF(prepdataframe(niskin_ds, variable), surface_fix = TRUE)
    niskin_interpolated_list[[variable]] <- temp_int %>%
      select(date, depth, value_int) %>%
      rename(!!variable := value_int)
  }
  
  # Combine into wide format
  niskin_profiles <- niskin_interpolated_list %>%
    reduce(full_join, by = c("date", "depth"))
  
  # Cache for future use
  saveRDS(niskin_profiles, cache_file)
  cat(sprintf("Saved Niskin profiles to cache: %s\n", cache_file))
  cat(sprintf("  Dates: %d, depths 0-%dm\n",
              length(unique(niskin_profiles$date)),
              max(niskin_profiles$depth)))
  
  return(niskin_profiles)
}

#' Load all profile data and merge with scenario classification
#'
#' Main data loading function that combines:
#' - HPLC depth profiles
#' - Niskin depth profiles
#' - Scenario classification (upwelling index, isotherm, euphotic depth)
#'
#' @return List with components: hplc, niskin, scenario, dates_summary
load_profile_data <- function() {
  
  # Load profiles
  hplc_profiles <- load_hplc_profiles()
  niskin_profiles <- load_niskin_profiles()
  
  # Load scenario classification
  scenario_data <- get_combined_scenario_data()
  
  # Summary of date coverage
  hplc_dates <- unique(hplc_profiles$date)
  niskin_dates <- unique(niskin_profiles$date)
  scenario_dates <- unique(scenario_data$date)
  
  all_dates <- unique(c(hplc_dates, niskin_dates, scenario_dates))
  
  dates_summary <- tibble(
    date = all_dates,
    has_hplc = date %in% hplc_dates,
    has_niskin = date %in% niskin_dates,
    has_scenario = date %in% scenario_dates
  ) %>%
    left_join(scenario_data %>% select(date, upwelling, Isotherm_21, euphotic_depth_obs),
              by = "date") %>%
    arrange(date)
  
  cat("\n=== Date Coverage Summary ===\n")
  cat(sprintf("  Total unique dates:     %d\n", length(all_dates)))
  cat(sprintf("  With HPLC data:         %d\n", sum(dates_summary$has_hplc)))
  cat(sprintf("  With Niskin data:       %d\n", sum(dates_summary$has_niskin)))
  cat(sprintf("  With scenario class:    %d\n", sum(dates_summary$has_scenario)))
  cat(sprintf("  With observed EuZ:      %d\n", sum(!is.na(dates_summary$euphotic_depth_obs))))
  
  return(list(
    hplc = hplc_profiles,
    niskin = niskin_profiles,
    scenario = scenario_data,
    dates_summary = dates_summary
  ))
}

# =============================================================================
# 2. EUPHOTIC DEPTH PROXY MODEL
# =============================================================================

#' Fit linear model to predict euphotic depth
#'
#' Supports two model types:
#' - "isotherm": EuZ ~ Isotherm_21 (original, simpler)
#' - "isotherm_chl": EuZ ~ Isotherm_21 + log10(Chl_integrated) (improved)
#'
#' The isotherm_chl model uses fluorometric chlorophyll integrated over 0-100m
#' from Niskin profiles, which substantially improves prediction accuracy
#' (adj_R² ~0.71 vs ~0.41 for isotherm-only).
#'
#' @param scenario_data Output from get_combined_scenario_data()
#' @param niskin_profiles Niskin depth profiles (required for "isotherm_chl" model)
#' @param model_type One of "isotherm" (default) or "isotherm_chl"
#' @param reference_depth Depth for chlorophyll integration (default 100m)
#' @return List with: model, model_type, coefficients, r_squared, predictions for all dates
fit_euphotic_proxy <- function(scenario_data, 
                                niskin_profiles = NULL,
                                model_type = "isotherm",
                                reference_depth = 100) {
  
  # Validate inputs
  if (!model_type %in% c("isotherm", "isotherm_chl")) {
    stop("model_type must be one of: 'isotherm', 'isotherm_chl'")
  }
  
  if (model_type == "isotherm_chl" && is.null(niskin_profiles)) {
    stop("niskin_profiles required for 'isotherm_chl' model type")
  }
  
  # =========================================================================
  # PREPARE DATA
  # =========================================================================
  
  # Start with dates that have observed euphotic depth
  model_data <- scenario_data %>%
    filter(!is.na(euphotic_depth_obs) & !is.na(Isotherm_21))
  
  # For isotherm_chl model, calculate integrated chlorophyll
  if (model_type == "isotherm_chl") {
    
    # Calculate integrated Chl (0-reference_depth) for all dates with Niskin data
    chl_integrated_all <- niskin_profiles %>%
      filter(depth <= reference_depth) %>%
      filter(!is.na(Chlorophyll)) %>%
      group_by(date) %>%
      summarize(
        Chl_integrated = sum(Chlorophyll, na.rm = TRUE),
        n_depths = n(),
        max_chl_depth = max(depth),
        .groups = "drop"
      ) %>%
      # Only keep profiles with reasonable coverage
      filter(n_depths >= 50) %>%
      # Add log-transformed Chl
      mutate(log_Chl_int = log10(Chl_integrated + 0.1))
    
    cat(sprintf("Chlorophyll integration (0-%dm):\n", reference_depth))
    cat(sprintf("  Dates with sufficient Chl data: %d\n", nrow(chl_integrated_all)))
    
    # Join to model data
    model_data <- model_data %>%
      left_join(chl_integrated_all %>% select(date, Chl_integrated, log_Chl_int), 
                by = "date") %>%
      filter(!is.na(Chl_integrated))
  }
  
  cat(sprintf("\nFitting EuZ proxy model (%s) on %d observations...\n", 
              model_type, nrow(model_data)))
  
  # =========================================================================
  # FIT MODEL
  # =========================================================================
  
  if (model_type == "isotherm") {
    # Original simple model
    model <- lm(euphotic_depth_obs ~ Isotherm_21, data = model_data)
    
    coefs <- list(
      intercept = coef(model)[1],
      Isotherm_21 = coef(model)[2]
    )
    
    cat(sprintf("  Model: EuZ = %.2f + %.4f × Isotherm_21\n", 
                coefs$intercept, coefs$Isotherm_21))
    
  } else if (model_type == "isotherm_chl") {
    # Improved model with log-transformed integrated chlorophyll
    model <- lm(euphotic_depth_obs ~ Isotherm_21 + log_Chl_int, data = model_data)
    
    coefs <- list(
      intercept = coef(model)[1],
      Isotherm_21 = coef(model)[2],
      log_Chl_int = coef(model)[3]
    )
    
    cat(sprintf("  Model: EuZ = %.2f + %.4f × Isotherm_21 + %.4f × log10(Chl_int)\n", 
                coefs$intercept, coefs$Isotherm_21, coefs$log_Chl_int))
    
    # Verify coefficient signs
    iso_sign <- ifelse(coefs$Isotherm_21 > 0, "✓", "✗")
    chl_sign <- ifelse(coefs$log_Chl_int < 0, "✓", "✗")
    cat(sprintf("  Coefficient signs: Isotherm %s (expect +), log(Chl) %s (expect -)\n",
                iso_sign, chl_sign))
  }
  
  r_squared <- summary(model)$r.squared
  adj_r_squared <- summary(model)$adj.r.squared
  rmse <- sqrt(mean(residuals(model)^2))
  
  cat(sprintf("  R-squared: %.3f (adj: %.3f)\n", r_squared, adj_r_squared))
  cat(sprintf("  RMSE: %.2f m\n", rmse))
  
  # =========================================================================
  # GENERATE PREDICTIONS FOR ALL DATES
  # =========================================================================
  
  if (model_type == "isotherm") {
    # Predict for all dates with isotherm data
    predictions <- scenario_data %>%
      filter(!is.na(Isotherm_21)) %>%
      mutate(
        euphotic_depth_predicted = predict(model, newdata = .),
        euphotic_depth_best = ifelse(!is.na(euphotic_depth_obs), 
                                      euphotic_depth_obs, 
                                      euphotic_depth_predicted),
        prediction_source = ifelse(!is.na(euphotic_depth_obs), "observed", "isotherm_model")
      ) %>%
      select(date, Isotherm_21, upwelling, euphotic_depth_obs, 
             euphotic_depth_predicted, euphotic_depth_best, prediction_source)
    
  } else if (model_type == "isotherm_chl") {
    # Need to join chlorophyll data for predictions
    predictions <- scenario_data %>%
      filter(!is.na(Isotherm_21)) %>%
      left_join(chl_integrated_all %>% select(date, Chl_integrated, log_Chl_int), 
                by = "date") %>%
      mutate(
        # Predict where we have both Isotherm and Chl
        euphotic_depth_predicted = ifelse(
          !is.na(log_Chl_int),
          predict(model, newdata = .),
          NA_real_
        ),
        # Best estimate: observed > predicted (with Chl) > fallback to isotherm-only
        euphotic_depth_best = case_when(
          !is.na(euphotic_depth_obs) ~ euphotic_depth_obs,
          !is.na(euphotic_depth_predicted) ~ euphotic_depth_predicted,
          TRUE ~ NA_real_
        ),
        prediction_source = case_when(
          !is.na(euphotic_depth_obs) ~ "observed",
          !is.na(euphotic_depth_predicted) ~ "isotherm_chl_model",
          TRUE ~ "no_prediction"
        )
      ) %>%
      select(date, Isotherm_21, Chl_integrated, upwelling, euphotic_depth_obs, 
             euphotic_depth_predicted, euphotic_depth_best, prediction_source)
  }
  
  # Summary of prediction coverage
  cat(sprintf("\nPrediction coverage:\n"))
  cat(sprintf("  Total dates with Isotherm_21: %d\n", nrow(predictions)))
  cat(sprintf("  With observed EuZ: %d\n", sum(predictions$prediction_source == "observed")))
  if (model_type == "isotherm") {
    cat(sprintf("  With predicted EuZ: %d\n", sum(predictions$prediction_source == "isotherm_model")))
  } else {
    cat(sprintf("  With predicted EuZ (iso+chl): %d\n", sum(predictions$prediction_source == "isotherm_chl_model")))
    cat(sprintf("  No prediction (missing Chl): %d\n", sum(predictions$prediction_source == "no_prediction")))
  }
  
  # =========================================================================
  # RETURN RESULTS
  # =========================================================================
  
  result <- list(
    model = model,
    model_type = model_type,
    coefficients = coefs,
    r_squared = r_squared,
    adj_r_squared = adj_r_squared,
    rmse = rmse,
    n_obs = nrow(model_data),
    predictions = predictions
  )
  
  # Add reference depth for isotherm_chl model
  if (model_type == "isotherm_chl") {
    result$reference_depth <- reference_depth
    result$chl_integrated_all <- chl_integrated_all
  }
  
  return(result)
}


#' Compare euphotic depth proxy models
#'
#' Fits both isotherm-only and isotherm+Chl models and returns comparison.
#'
#' @param scenario_data Output from get_combined_scenario_data()
#' @param niskin_profiles Niskin depth profiles
#' @param reference_depth Depth for chlorophyll integration (default 100m)
#' @return List with both models and comparison statistics
compare_euphotic_proxy_models <- function(scenario_data, 
                                           niskin_profiles,
                                           reference_depth = 100) {
  
  cat("=== Comparing Euphotic Depth Proxy Models ===\n\n")
  
  # Fit both models
  cat("--- Model 1: Isotherm only ---\n")
  model_iso <- fit_euphotic_proxy(scenario_data, model_type = "isotherm")
  
  cat("\n--- Model 2: Isotherm + log(Chl) ---\n")
  model_iso_chl <- fit_euphotic_proxy(scenario_data, niskin_profiles, 
                                       model_type = "isotherm_chl",
                                       reference_depth = reference_depth)
  
  # Comparison summary
  cat("\n=== Model Comparison Summary ===\n")
  comparison <- tibble(
    model = c("Isotherm only", "Isotherm + log(Chl)"),
    n = c(model_iso$n_obs, model_iso_chl$n_obs),
    R2 = c(model_iso$r_squared, model_iso_chl$r_squared),
    adj_R2 = c(model_iso$adj_r_squared, model_iso_chl$adj_r_squared),
    RMSE = c(model_iso$rmse, model_iso_chl$rmse)
  )
  print(comparison)
  
  cat(sprintf("\nImprovement with Chl:\n"))
  cat(sprintf("  adj_R² increase: +%.3f (%.1f%% relative)\n", 
              model_iso_chl$adj_r_squared - model_iso$adj_r_squared,
              100 * (model_iso_chl$adj_r_squared - model_iso$adj_r_squared) / model_iso$adj_r_squared))
  cat(sprintf("  RMSE reduction: -%.2f m (%.1f%% relative)\n",
              model_iso$rmse - model_iso_chl$rmse,
              100 * (model_iso$rmse - model_iso_chl$rmse) / model_iso$rmse))
  
  return(list(
    isotherm = model_iso,
    isotherm_chl = model_iso_chl,
    comparison = comparison
  ))
}


# =========================================================================
# 3. ASSIGN DEPTH CUTOFFS
# =========================================================================

if (depth_mode == "fixed") {
    backbone$depth_cutoff <- fixed_depth
    backbone$cutoff_source <- "fixed"

} else if (depth_mode == "scenario") {
    backbone <- backbone %>%
      mutate(
        depth_cutoff = case_when(
          upwelling == "upwelling" ~ scenario_depths["upwelling"],
          upwelling == "relaxed" ~ scenario_depths["relaxed"],
          TRUE ~ NA_real_
        ),
        cutoff_source = "scenario"
      )

} else if (depth_mode == "dynamic") {
    if (is.null(proxy_model)) {
      stop("proxy_model required for dynamic depth mode")
    }
    
    # Use pre-computed predictions from proxy_model
    # This works for both "isotherm" and "isotherm_chl" model types
    proxy_predictions <- proxy_model$predictions %>%
      mutate(time_month = format(date, "%m-%Y")) %>%
      group_by(time_month) %>%
      summarize(
        euphotic_depth_predicted = mean(euphotic_depth_best, na.rm = TRUE),
        prediction_source = first(prediction_source),
        .groups = "drop"
      ) %>%
      mutate(euphotic_depth_predicted = ifelse(is.nan(euphotic_depth_predicted), 
                                                NA_real_, 
                                                euphotic_depth_predicted))
    
    backbone <- backbone %>%
      left_join(proxy_predictions %>% select(time_month, euphotic_depth_predicted, prediction_source),
                by = "time_month") %>%
      mutate(
        depth_cutoff = euphotic_depth_predicted,
        cutoff_source = ifelse(!is.na(prediction_source), prediction_source, "no_prediction")
      ) %>%
      select(-euphotic_depth_predicted, -prediction_source)
    
    cat(sprintf("  Dynamic cutoffs from %s model\n", proxy_model$model_type))

} else if (depth_mode == "observed") {
    # Use actual observed euphotic depth where available
    backbone <- backbone %>%
      mutate(
        depth_cutoff = euphotic_depth_obs,
        cutoff_source = "observed"
      )
}

cat(sprintf("  With depth cutoff: %d months\n", sum(!is.na(backbone$depth_cutoff))))

#' Integrate profiles to specified depth cutoffs
#'
#' @param profile_data Output from load_profile_data()
#' @param depth_mode One of "fixed", "scenario", or "dynamic"
#' @param fixed_depth Depth for fixed mode
#' @param scenario_depths Named vector for scenario mode
#' @param proxy_model Output from fit_euphotic_proxy() for dynamic mode
#' @param na_threshold Max allowed NAs in integration interval (per variable)
#' @return List with integrated HPLC and Niskin data
integrate_to_depth <- function(profile_data,
                                depth_mode = "fixed",
                                fixed_depth = 50,
                                scenario_depths = c(upwelling = 35, relaxed = 50),
                                proxy_model = NULL,
                                na_threshold = 10) {
  
  cat(sprintf("\n=== Integrating profiles (mode: %s) ===\n", depth_mode))
  
  # Get cutoffs for all dates
  all_dates <- unique(c(profile_data$hplc$date, profile_data$niskin$date))
  cutoffs <- get_depth_cutoffs(all_dates, profile_data$scenario, 
                                depth_mode, fixed_depth, scenario_depths, proxy_model)
  
  # --- Integrate HPLC ---
  hplc_integrated <- integrate_hplc(profile_data$hplc, cutoffs, na_threshold)
  
  # --- Integrate Niskin ---
  niskin_integrated <- integrate_niskin(profile_data$niskin, cutoffs, na_threshold)
  
  # Add scenario info to both
  hplc_integrated <- hplc_integrated %>%
    left_join(profile_data$scenario %>% select(date, upwelling, ui, Isotherm_21, sst),
              by = "date")
  
  niskin_integrated <- niskin_integrated %>%
    left_join(profile_data$scenario %>% select(date, upwelling, ui, Isotherm_21, sst),
              by = "date")
  
  cat(sprintf("  HPLC: %d dates integrated\n", nrow(hplc_integrated)))
  cat(sprintf("  Niskin: %d dates integrated\n", nrow(niskin_integrated)))
  
  return(list(
    hplc = hplc_integrated,
    niskin = niskin_integrated,
    cutoffs = cutoffs,
    depth_mode = depth_mode
  ))
}

#' Integrate HPLC profiles and calculate size fractions
#'
#' @param hplc_profiles HPLC depth profile data
#' @param cutoffs Data frame with date and depth_cutoff
#' @param na_threshold Max NAs allowed
#' @return Data frame with integrated pigments and size fractions
integrate_hplc <- function(hplc_profiles, cutoffs, na_threshold = 10) {
  
  pigments <- c("Pras", "Lut", "Fuco", "Perid", "Allo", "But_fuco",
                "Hex_fuco", "Zea", "Tot_Chl_b", "DP", "Tot_Chl_a",
                "TChl", "Chl_c1c2", "Chl_c3")
  
  # Join cutoffs and filter/integrate
  hplc_integrated <- hplc_profiles %>%
    left_join(cutoffs %>% select(date, depth_cutoff), by = "date") %>%
    filter(!is.na(depth_cutoff)) %>%
    filter(depth >= 0 & depth <= depth_cutoff) %>%
    group_by(date, depth_cutoff) %>%
    summarize(
      across(all_of(pigments), 
             list(
               integrated = ~ifelse(sum(is.na(.x)) < na_threshold, 
                                    mean(.x, na.rm = TRUE) * first(depth_cutoff), 
                                    NA_real_),
               mean = ~ifelse(sum(is.na(.x)) < na_threshold,
                              mean(.x, na.rm = TRUE),
                              NA_real_)
             )),
      n_depths = n(),
      .groups = "drop"
    )
  
  # Calculate size fractions from integrated pigments
  hplc_integrated <- hplc_integrated %>%
    mutate(
      # Diagnostic pigment sum (using integrated values)
      DP2 = 1.41 * Fuco_integrated + 1.41 * Perid_integrated + 
            0.60 * Allo_integrated + 0.35 * But_fuco_integrated + 
            1.27 * Hex_fuco_integrated + 0.86 * Zea_integrated + 
            1.01 * Tot_Chl_b_integrated,
      DP2 = ifelse(DP2 < 0.001, NA, DP2),
      
      # Relative size fractions
      micro = (1.41 * Fuco_integrated + 1.41 * Perid_integrated) / DP2,
      nano  = (0.60 * Allo_integrated + 0.35 * But_fuco_integrated + 
               1.27 * Hex_fuco_integrated) / DP2,
      pico  = (0.86 * Zea_integrated + 1.01 * Tot_Chl_b_integrated) / DP2,
      
      # Absolute biomass (integrated Chl a partitioned by size)
      micro_abs = micro * Tot_Chl_a_integrated,
      nano_abs  = nano * Tot_Chl_a_integrated,
      pico_abs  = pico * Tot_Chl_a_integrated,
      
      # Size spectral metrics
      size_centroid = micro * log10(63) + nano * log10(6.3) + pico * log10(0.63),
      
      size_shannon = -(ifelse(micro > 0, micro * log(micro), 0) +
                       ifelse(nano > 0, nano * log(nano), 0) +
                       ifelse(pico > 0, pico * log(pico), 0)),
      
      nbss_slope = ifelse(
        micro_abs > 0 & pico_abs > 0,
        (log10(micro_abs) - log10(pico_abs)) / (log10(63) - log10(0.63)),
        NA_real_
      )
    )
  
  return(hplc_integrated)
}

#' Integrate Niskin profiles
#'
#' @param niskin_profiles Niskin depth profile data
#' @param cutoffs Data frame with date and depth_cutoff
#' @param na_threshold Max NAs allowed
#' @return Data frame with integrated/averaged Niskin variables
integrate_niskin <- function(niskin_profiles, cutoffs, na_threshold = 10) {
  
  niskin_vars <- c("NO3_merged", "Chlorophyll", "PrimaryProductivity", 
                   "PN_ug_L", "Temperature")
  
  # Variables to integrate (flux-like)
  vars_to_integrate <- c("PrimaryProductivity", "Chlorophyll")
  # Variables to average (concentration-like)
  vars_to_average <- c("NO3_merged", "PN_ug_L", "Temperature")
  
  niskin_integrated <- niskin_profiles %>%
    left_join(cutoffs %>% select(date, depth_cutoff), by = "date") %>%
    filter(!is.na(depth_cutoff)) %>%
    filter(depth >= 0 & depth <= depth_cutoff) %>%
    group_by(date, depth_cutoff) %>%
    summarize(
      # Integrated variables
      across(all_of(vars_to_integrate),
             ~ifelse(sum(is.na(.x)) < na_threshold,
                     mean(.x, na.rm = TRUE) * first(depth_cutoff),
                     NA_real_),
             .names = "{.col}_integrated"),
      # Averaged variables
      across(all_of(vars_to_average),
             ~ifelse(sum(is.na(.x)) < na_threshold,
                     mean(.x, na.rm = TRUE),
                     NA_real_),
             .names = "{.col}_mean"),
      n_depths = n(),
      .groups = "drop"
    ) %>%
    # Convert PON units: µg/L -> mmol N m-3
    mutate(PON_mmol = PN_ug_L_mean / 14.007)
  
  return(niskin_integrated)
}

# =============================================================================
# 4. COMPARISON HELPERS
# =============================================================================

#' Compare integration strategies side by side
#'
#' Runs all three integration modes and returns combined results
#'
#' @param profile_data Output from load_profile_data()
#' @param proxy_model Output from fit_euphotic_proxy()
#' @param fixed_depth Depth for fixed mode
#' @param scenario_depths Named vector for scenario mode
#' @return List with all three integration results and comparison summaries
compare_integration_strategies <- function(profile_data, 
                                            proxy_model,
                                            fixed_depth = 50,
                                            scenario_depths = c(upwelling = 35, relaxed = 50)) {
  
  cat("\n========================================\n")
  cat("Comparing Integration Strategies\n")
  cat("========================================\n")
  
  # Run all three modes
  int_fixed <- integrate_to_depth(profile_data, depth_mode = "fixed", 
                                   fixed_depth = fixed_depth)
  
  int_scenario <- integrate_to_depth(profile_data, depth_mode = "scenario",
                                      scenario_depths = scenario_depths)
  
  int_dynamic <- integrate_to_depth(profile_data, depth_mode = "dynamic",
                                     proxy_model = proxy_model)
  
  # Tag each with strategy name
  int_fixed$hplc$strategy <- "fixed"
  int_scenario$hplc$strategy <- "scenario"
  int_dynamic$hplc$strategy <- "dynamic"
  
  int_fixed$niskin$strategy <- "fixed"
  int_scenario$niskin$strategy <- "scenario"
  int_dynamic$niskin$strategy <- "dynamic"
  
  # Combine for comparison
  hplc_comparison <- bind_rows(
    int_fixed$hplc,
    int_scenario$hplc,
    int_dynamic$hplc
  )
  
  niskin_comparison <- bind_rows(
    int_fixed$niskin,
    int_scenario$niskin,
    int_dynamic$niskin
  )
  
  return(list(
    fixed = int_fixed,
    scenario = int_scenario,
    dynamic = int_dynamic,
    hplc_comparison = hplc_comparison,
    niskin_comparison = niskin_comparison
  ))
}


# =============================================================================
# 5. FULL SCENARIO DATA WITH ALL VARIABLES
# =============================================================================

# Unit conversion constants (matching Python cariaco_obs.py)
C_TO_CHL     <- 50.0      # mg C : mg Chl
C_TO_DW      <- 0.4       # mg C : mg DW (zooplankton)
REDFIELD_N_C <- 16 / 106  # mmol N : mmol C
MW_CARBON    <- 12.01     # g mol⁻¹
MW_N         <- 14.007    # g mol⁻¹

#' Convert chlorophyll to nitrogen units
#' @param chl_integrated Integrated chlorophyll (mg Chl m⁻²)
#' @param depth_cutoff Integration depth (m)
#' @return Concentration in mmol N m⁻³
chl_to_mmolN <- function(chl_integrated, depth_cutoff) {
  # mg Chl m⁻² → mg Chl m⁻³ → mg C m⁻³ → mmol C m⁻³ → mmol N m⁻³
  (chl_integrated / depth_cutoff) * C_TO_CHL / MW_CARBON * REDFIELD_N_C
}

#' Convert zooplankton dry weight to nitrogen units
#' @param biomass_dw Dry weight biomass (mg DW m⁻³)
#' @return Concentration in mmol N m⁻³
zoo_dw_to_mmolN <- function(biomass_dw) {
  # mg DW m⁻³ → mg C m⁻³ → mmol C m⁻³ → mmol N m⁻³
  biomass_dw * C_TO_DW / MW_CARBON * REDFIELD_N_C
}

#' Load zooplankton data
#' @return Data frame with date and biomass columns
load_zoo_data <- function() {
  zoo <- readRDS("../processed/Zoo_processed.rds")
  
  # Ensure date column exists
  if (!"date" %in% names(zoo)) {
    stop("Zoo_processed.rds must have a 'date' column")
  }
  
  zoo %>%
    select(date, BIOMASS_200, BIOMASS_500, AFDW_200, AFDW_500) %>%
    # Convert to mmol N m⁻³
    mutate(
      zoo_gt200_mmolN = zoo_dw_to_mmolN(BIOMASS_200),
      zoo_gt500_mmolN = zoo_dw_to_mmolN(BIOMASS_500)
    )
}

#' Load sediment trap data
#' @param trap_lag_months Lag in months (0 = same month, 1 = trap month after scenario month)
#' @return Data frame with time_month and flux columns
load_trap_data <- function(trap_lag_months = 0) {
  trap <- readRDS("../processed/SedTrap_monthly.rds")
  
  # Filter to 225m trap (primary depth)
  trap_225 <- trap %>%
    filter(depth_trap == 225) %>%
    select(time_month, MF_N_mmol, MF_Corg_mmol, CN_ratio, n_samples)
  
  # Apply lag if specified
  if (trap_lag_months != 0) {
    # Parse time_month (MM-YYYY format)
    trap_225 <- trap_225 %>%
      mutate(
        month_num = as.numeric(substr(time_month, 1, 2)),
        year_num = as.numeric(substr(time_month, 4, 7)),
        # Shift back by lag (so trap data aligns with earlier scenario)
        adj_month = month_num - trap_lag_months,
        adj_year = year_num + floor((adj_month - 1) / 12),
        adj_month = ((adj_month - 1) %% 12) + 1,
        time_month_adj = sprintf("%02d-%04d", adj_month, adj_year)
      ) %>%
      select(-month_num, -year_num, -adj_month, -adj_year) %>%
      rename(time_month_original = time_month, time_month = time_month_adj)
  }
  
  return(trap_225)
}



#' Get full scenario data with all variables in model units
#'
#' Uses a COMPLETE MONTHLY GRID as backbone to preserve maximum coverage.
#' All data sources are left-joined, so missing data shows as NA.
#'
#' @param profile_data Output from load_profile_data()
#' @param depth_mode One of "fixed", "scenario", or "dynamic"
#' @param fixed_depth Depth for fixed mode (default 50)
#' @param scenario_depths Named vector for scenario mode (default c(upwelling=35, relaxed=50))
#' @param proxy_model Output from fit_euphotic_proxy() for dynamic mode
#' @param trap_lag_months Lag for sediment trap matching (default 0)
#' @param start_date Start of time series (default "1995-11-01")
#' @param end_date End of time series (default "2017-01-01")
#' @return Data frame with per-month values, all in model units
get_full_scenario_data <- function(profile_data,
                                    depth_mode = "scenario",
                                    fixed_depth = 50,
                                    scenario_depths = c(upwelling = 35, relaxed = 50),
                                    proxy_model = NULL,
                                    trap_lag_months = 0,
                                    start_date = as.Date("1995-11-01"),
                                    end_date = as.Date("2017-01-01")) {
  
  cat(sprintf("\n=== Building full scenario dataset (mode: %s) ===\n", depth_mode))
  
  # =========================================================================
  # 1. CREATE COMPLETE MONTHLY BACKBONE
  # =========================================================================
  
  backbone <- data.frame(
    date = seq(from = start_date, to = end_date, by = "month")
  ) %>%
    mutate(
      time_month = format(date, format = "%m-%Y"),
      year = as.numeric(format(date, "%Y")),
      month = as.numeric(format(date, "%m"))
    )
  
  cat(sprintf("Complete monthly backbone: %d months\n", nrow(backbone)))
  cat(sprintf("  From %s to %s\n", start_date, end_date))
  
  # =========================================================================
  # 2. JOIN SCENARIO CLASSIFICATION (CTD-derived)
  # =========================================================================
  
  # Aggregate scenario data to monthly (in case of multiple samples per month)
  scenario_monthly <- profile_data$scenario %>%
    mutate(time_month = format(date, "%m-%Y")) %>%
    group_by(time_month) %>%
    summarize(
      upwelling = first(na.omit(upwelling)),
      ui = first(na.omit(ui)),
      Isotherm_21 = mean(Isotherm_21, na.rm = TRUE),
      sst = mean(sst, na.rm = TRUE),
      temp_50m = mean(temp_50m, na.rm = TRUE),
      euphotic_depth_obs = mean(euphotic_depth_obs, na.rm = TRUE),
      MLD = mean(MLD, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    # Replace NaN with NA
    mutate(across(where(is.numeric), ~ifelse(is.nan(.), NA_real_, .)))
  
  backbone <- backbone %>%
    left_join(scenario_monthly, by = "time_month")
  
  cat(sprintf("  With upwelling class: %d months\n", sum(!is.na(backbone$upwelling))))
  
  # =========================================================================
  # 3. ASSIGN DEPTH CUTOFFS
  # =========================================================================
  
  if (depth_mode == "fixed") {
    backbone$depth_cutoff <- fixed_depth
    backbone$cutoff_source <- "fixed"
    
  } else if (depth_mode == "scenario") {
    backbone <- backbone %>%
      mutate(
        depth_cutoff = case_when(
          upwelling == "upwelling" ~ scenario_depths["upwelling"],
          upwelling == "relaxed" ~ scenario_depths["relaxed"],
          TRUE ~ NA_real_
        ),
        cutoff_source = "scenario"
      )
    
  } else if (depth_mode == "dynamic") {
    if (is.null(proxy_model)) {
      stop("proxy_model required for dynamic depth mode")
    }
    # Use predicted euphotic depth where available
    backbone <- backbone %>%
      mutate(
        depth_cutoff = ifelse(
          !is.na(euphotic_depth_obs),
          euphotic_depth_obs,
          proxy_model$intercept + proxy_model$slope * Isotherm_21
        ),
        cutoff_source = "dynamic"
      )
  } else if (depth_mode == "observed") {
    # Use actual observed euphotic depth where available
    backbone <- backbone %>%
      mutate(
        depth_cutoff = euphotic_depth_obs,
        cutoff_source = "observed"
      )
  }
  
  cat(sprintf("  With depth cutoff: %d months\n", sum(!is.na(backbone$depth_cutoff))))
  
  # =========================================================================
  # 4. INTEGRATE HPLC DATA (aggregate to monthly)
  # =========================================================================
  
  # First integrate each DATE to its depth cutoff, then aggregate to monthly
  hplc_with_cutoff <- profile_data$hplc %>%
    mutate(time_month = format(date, "%m-%Y")) %>%
    left_join(backbone %>% select(time_month, depth_cutoff), by = "time_month")
  
  hplc_integrated <- hplc_with_cutoff %>%
    filter(!is.na(depth_cutoff)) %>%
    filter(depth >= 0 & depth <= depth_cutoff) %>%
    group_by(date, time_month, depth_cutoff) %>%
    summarize(
      across(c(Fuco, Perid, Allo, But_fuco, Hex_fuco, Zea, Tot_Chl_b, Tot_Chl_a),
             ~ifelse(sum(is.na(.x)) < 10, 
                     mean(.x, na.rm = TRUE) * first(depth_cutoff), 
                     NA_real_),
             .names = "{.col}_int"),
      .groups = "drop"
    ) %>%
    # Calculate size fractions per date
    mutate(
      DP2 = 1.41 * Fuco_int + 1.41 * Perid_int + 
            0.60 * Allo_int + 0.35 * But_fuco_int + 
            1.27 * Hex_fuco_int + 0.86 * Zea_int + 
            1.01 * Tot_Chl_b_int,
      DP2 = ifelse(DP2 < 0.001, NA, DP2),
      
      micro = (1.41 * Fuco_int + 1.41 * Perid_int) / DP2,
      nano  = (0.60 * Allo_int + 0.35 * But_fuco_int + 1.27 * Hex_fuco_int) / DP2,
      pico  = (0.86 * Zea_int + 1.01 * Tot_Chl_b_int) / DP2,
      
      micro_abs = micro * Tot_Chl_a_int,
      nano_abs  = nano * Tot_Chl_a_int,
      pico_abs  = pico * Tot_Chl_a_int,
      
      # Convert to model units (mmol N m⁻³)
      # Total Chlorophyll a: directly from Tot_Chl_a
      TotChlA_mmolN = chl_to_mmolN(Tot_Chl_a_int, depth_cutoff),
        
      # Size classes: partitioned from total
      micro_mmolN = chl_to_mmolN(micro_abs, depth_cutoff),
      nano_mmolN  = chl_to_mmolN(nano_abs, depth_cutoff),
      pico_mmolN  = chl_to_mmolN(pico_abs, depth_cutoff),
      
      size_centroid = micro * log10(63) + nano * log10(6.3) + pico * log10(0.63),
      size_shannon = -(ifelse(micro > 0, micro * log(micro), 0) +
                       ifelse(nano > 0, nano * log(nano), 0) +
                       ifelse(pico > 0, pico * log(pico), 0)),
      nbss_slope = ifelse(
        micro_abs > 0 & pico_abs > 0,
        (log10(micro_abs) - log10(pico_abs)) / (log10(63) - log10(0.63)),
        NA_real_
      )
    )
  
  # Aggregate to monthly
  hplc_monthly <- hplc_integrated %>%
    group_by(time_month) %>%
    summarize(
      # Total Chlorophyll a
      TotChlA_mmolN = mean(TotChlA_mmolN, na.rm = TRUE),
      
      # Size classes
      micro_mmolN = mean(micro_mmolN, na.rm = TRUE),
      nano_mmolN = mean(nano_mmolN, na.rm = TRUE),
      pico_mmolN = mean(pico_mmolN, na.rm = TRUE),
        
      micro_frac = mean(micro, na.rm = TRUE),
      nano_frac = mean(nano, na.rm = TRUE),
      pico_frac = mean(pico, na.rm = TRUE),
      size_centroid = mean(size_centroid, na.rm = TRUE),
      size_shannon = mean(size_shannon, na.rm = TRUE),
      nbss_slope = mean(nbss_slope, na.rm = TRUE),
      n_hplc_samples = n(),
      .groups = "drop"
    ) %>%
    mutate(across(where(is.numeric), ~ifelse(is.nan(.), NA_real_, .)))
  
  cat(sprintf("HPLC: %d months with data\n", nrow(hplc_monthly)))
    
  
  # =========================================================================
  # 5. INTEGRATE NISKIN DATA (aggregate to monthly)
  # =========================================================================
  
  niskin_with_cutoff <- profile_data$niskin %>%
    mutate(time_month = format(date, "%m-%Y")) %>%
    left_join(backbone %>% select(time_month, depth_cutoff), by = "time_month")
  
  niskin_integrated <- niskin_with_cutoff %>%
    filter(!is.na(depth_cutoff)) %>%
    filter(depth >= 0 & depth <= depth_cutoff) %>%
    group_by(date, time_month, depth_cutoff) %>%
    summarize(
      NO3_mmolN = ifelse(sum(is.na(NO3_merged)) < 10,
                         mean(NO3_merged, na.rm = TRUE), NA_real_),
      PON_mmolN = ifelse(sum(is.na(PN_ug_L)) < 10,
                         mean(PN_ug_L, na.rm = TRUE) / 14.007, NA_real_),
      PP_mgC_m2_d = ifelse(sum(is.na(PrimaryProductivity)) < 10,
                           mean(PrimaryProductivity, na.rm = TRUE) * first(depth_cutoff) * 12,
                           NA_real_),
      Temp_C = ifelse(sum(is.na(Temperature)) < 10,
                      mean(Temperature, na.rm = TRUE), NA_real_),
      .groups = "drop"
    )
  
  # Aggregate to monthly
  niskin_monthly <- niskin_integrated %>%
    group_by(time_month) %>%
    summarize(
      NO3_mmolN = mean(NO3_mmolN, na.rm = TRUE),
      PON_mmolN = mean(PON_mmolN, na.rm = TRUE),
      PP_mgC_m2_d = mean(PP_mgC_m2_d, na.rm = TRUE),
      Temp_C = mean(Temp_C, na.rm = TRUE),
      n_niskin_samples = n(),
      .groups = "drop"
    ) %>%
    mutate(across(where(is.numeric), ~ifelse(is.nan(.), NA_real_, .)))
  
  cat(sprintf("Niskin: %d months with data\n", nrow(niskin_monthly)))
  
  # =========================================================================
  # 6. LOAD ZOOPLANKTON (aggregate to monthly)
  # =========================================================================
  
  zoo_monthly <- tryCatch({
    zoo <- readRDS("../processed/Zoo_processed.rds")
    zoo %>%
      mutate(time_month = format(date, "%m-%Y")) %>%
      group_by(time_month) %>%
      summarize(
        zoo_gt200_mmolN = mean(zoo_dw_to_mmolN(BIOMASS_200), na.rm = TRUE),
        zoo_gt500_mmolN = mean(zoo_dw_to_mmolN(BIOMASS_500), na.rm = TRUE),
        n_zoo_samples = n(),
        .groups = "drop"
      ) %>%
      mutate(across(where(is.numeric), ~ifelse(is.nan(.), NA_real_, .)))
  }, error = function(e) {
    cat("  Warning: Could not load zooplankton data:", e$message, "\n")
    NULL
  })
  
  if (!is.null(zoo_monthly)) {
    cat(sprintf("Zooplankton: %d months with data\n", nrow(zoo_monthly)))
  }
  
  # =========================================================================
  # 7. LOAD SEDIMENT TRAP (already monthly)
  # =========================================================================
  
  trap_monthly <- tryCatch({
    trap <- readRDS("../processed/SedTrap_monthly.rds")
    
    trap_225 <- trap %>%
      filter(depth_trap == 225) %>%
      select(time_month, MF_N_mmol, MF_Corg_mmol, CN_ratio)
    
    # Apply lag if specified
    if (trap_lag_months != 0) {
      trap_225 <- trap_225 %>%
        mutate(
          month_num = as.numeric(substr(time_month, 1, 2)),
          year_num = as.numeric(substr(time_month, 4, 7)),
          adj_month = month_num - trap_lag_months,
          adj_year = year_num + floor((adj_month - 1) / 12),
          adj_month = ((adj_month - 1) %% 12) + 1,
          time_month = sprintf("%02d-%04d", adj_month, adj_year)
        ) %>%
        select(time_month, MF_N_mmol, MF_Corg_mmol, CN_ratio)
    }
    
    trap_225 %>%
      rename(
        export_flux_mmolN = MF_N_mmol,
        export_flux_C = MF_Corg_mmol,
        trap_CN = CN_ratio
      )
  }, error = function(e) {
    cat("  Warning: Could not load sediment trap data:", e$message, "\n")
    NULL
  })
  
  if (!is.null(trap_monthly)) {
    cat(sprintf("Sediment trap: %d months with data (lag = %d)\n", 
                nrow(trap_monthly), trap_lag_months))
  }
  
  # =========================================================================
  # 8. JOIN ALL TO BACKBONE
  # =========================================================================
  
  full_data <- backbone %>%
    left_join(hplc_monthly, by = "time_month") %>%
    left_join(niskin_monthly, by = "time_month")
  
  if (!is.null(zoo_monthly)) {
    full_data <- full_data %>%
      left_join(zoo_monthly, by = "time_month")
  }
  
  if (!is.null(trap_monthly)) {
    full_data <- full_data %>%
      left_join(trap_monthly, by = "time_month")
  }
  
  full_data <- full_data %>%
    arrange(date)
  
  # Store metadata as attributes
  attr(full_data, "depth_mode") <- depth_mode
  attr(full_data, "trap_lag_months") <- trap_lag_months
  attr(full_data, "start_date") <- start_date
  attr(full_data, "end_date") <- end_date
  
  # =========================================================================
  # 9. COVERAGE SUMMARY
  # =========================================================================
  
  cat("\n=== Final Data Coverage ===\n")
  cat(sprintf("  Total months (backbone): %d\n", nrow(full_data)))
  cat(sprintf("  With upwelling class: %d\n", sum(!is.na(full_data$upwelling))))
  cat(sprintf("  With depth cutoff: %d\n", sum(!is.na(full_data$depth_cutoff))))
  cat(sprintf("  With TotChlA: %d\n", sum(!is.na(full_data$TotChlA_mmolN))))
  cat(sprintf("  With phyto size data: %d\n", sum(!is.na(full_data$micro_mmolN))))
  cat(sprintf("  With NO3: %d\n", sum(!is.na(full_data$NO3_mmolN))))
  cat(sprintf("  With PP: %d\n", sum(!is.na(full_data$PP_mgC_m2_d))))
  
  if (!is.null(zoo_monthly)) {
    cat(sprintf("  With zoo >200µm: %d\n", sum(!is.na(full_data$zoo_gt200_mmolN))))
  }
  if (!is.null(trap_monthly)) {
    cat(sprintf("  With export flux: %d\n", sum(!is.na(full_data$export_flux_mmolN))))
  }
  
  return(full_data)
}





#' Summarize full scenario data by upwelling class
#'
#' @param full_data Output from get_full_scenario_data()
#' @return Summary statistics grouped by upwelling
summarize_full_scenario <- function(full_data) {
  
  full_data %>%
    filter(!is.na(upwelling)) %>%
    group_by(upwelling) %>%
    summarize(
      n = n(),
      depth_cutoff = mean(depth_cutoff, na.rm = TRUE),
      
      # Phytoplankton (mmol N m⁻³)
      micro = mean(micro_mmolN, na.rm = TRUE),
      nano = mean(nano_mmolN, na.rm = TRUE),
      pico = mean(pico_mmolN, na.rm = TRUE),
      TotChlA = mean(TotChlA_mmolN, na.rm = TRUE),
      
      # Nutrients (mmol N m⁻³)
      NO3 = mean(NO3_mmolN, na.rm = TRUE),
      PON = mean(PON_mmolN, na.rm = TRUE),
      
      # Zooplankton (mmol N m⁻³)
      zoo_gt200 = mean(zoo_gt200_mmolN, na.rm = TRUE),
      zoo_gt500 = mean(zoo_gt500_mmolN, na.rm = TRUE),
      
      # Rates
      PP = mean(PP_mgC_m2_d, na.rm = TRUE),
      export = mean(export_flux_mmolN, na.rm = TRUE),
      
      # Environment
      Temp = mean(Temp_C, na.rm = TRUE),
      Isotherm_21 = mean(Isotherm_21, na.rm = TRUE),
      
      .groups = "drop"
    )
}




# =============================================================================
# 6. DOCUMENTATION AND METADATA
# =============================================================================

#' Get metadata describing all variables in full scenario data
#'
#' @return Tibble with variable descriptions, units, and methodology
get_scenario_metadata <- function() {
  tribble(
    ~variable,            ~units,              ~description,                                      ~methodology,
    # Identifiers
    "date",               "Date",              "Sampling date",                                   "From original datasets",
    "depth_cutoff",       "m",                 "Integration depth for this date",                 "Fixed (50m), scenario-specific (35/50m), or dynamic (EuZ proxy)",
    "upwelling",          "category",          "Upwelling regime classification",                 "Based on T at 50m: ≤22°C = upwelling, >22°C = relaxed",
    "ui",                 "category",          "Detailed upwelling index",
    "Based on T at 50m: ≤20°C strong, ≤21°C moderate, ≤22°C weak, >22°C relaxed",
    
    # Phytoplankton (HPLC-derived)
# Phytoplankton (HPLC-derived)
    "TotChlA_mmolN",      "mmol N m⁻³",        "Total Chlorophyll a",                             "HPLC Tot_Chl_a integrated to depth_cutoff, converted: (mg Chl / depth) × 50 / 12.01 × (16/106)",
    "micro_mmolN",        "mmol N m⁻³",        "Microphytoplankton (>20 µm) Chl a",               "Tot_Chl_a partitioned by diagnostic pigment fractions, then converted to mmol N",
    "nano_mmolN",         "mmol N m⁻³",        "Nanophytoplankton (2-20 µm) Chl a",               "Same as micro_mmolN",
    "pico_mmolN",         "mmol N m⁻³",        "Picophytoplankton (<2 µm) Chl a",                 "Same as micro_mmolN",
    "micro_frac",         "dimensionless",     "Microphytoplankton fraction of total",            "From diagnostic pigment ratios",
    "nano_frac",          "dimensionless",     "Nanophytoplankton fraction of total",             "From diagnostic pigment ratios",
    "pico_frac",          "dimensionless",     "Picophytoplankton fraction of total",             "From diagnostic pigment ratios",
    
    # Nutrients (Niskin-derived)
    "NO3_mmolN",          "mmol N m⁻³",        "Nitrate concentration (0 to depth_cutoff mean)",  "Interpolated Niskin NO3, mean over 0-depth_cutoff",
    "PON_mmolN",          "mmol N m⁻³",        "Particulate organic nitrogen",                    "Niskin PN (µg/L) / 14.007",
    
    # Rates
    "PP_mgC_m2_d",        "mg C m⁻² d⁻¹",      "Primary productivity (daily, integrated)",        "Niskin PP integrated to depth_cutoff, ×12 for daily (12h daylight)",
    "Temp_C",             "°C",                "Temperature (0 to depth_cutoff mean)",            "Interpolated Niskin temperature",
    
    # Zooplankton
    "zoo_gt200_mmolN",    "mmol N m⁻³",        "Zooplankton biomass >200 µm",                     "Bongo net 0-200m, mg DW × 0.4 / 12.01 × (16/106). NOTE: Net depth (200m) differs from euphotic depth",
    "zoo_gt500_mmolN",    "mmol N m⁻³",        "Zooplankton biomass >500 µm",                     "Same as >200 µm, different mesh",
    
    # Export
    "export_flux_mmolN",  "mmol N m⁻² d⁻¹",    "Particulate N export flux",                       "Sediment trap at 225m, duration-weighted monthly mean. Raw flux, no Martin correction applied",
    "trap_CN",            "mol:mol",           "C:N ratio of sinking particles",                  "From sediment trap Corg and N fluxes",
    
    # Environment
    "Isotherm_21",        "m",                 "Depth of 21°C isotherm",                          "From CTD, first depth where T < 21°C",
    "sst",                "°C",                "Sea surface temperature",                         "Mean temperature 0-10m from CTD"
  )
}

#' Summarize full scenario data with complete statistics
#'
#' Reports mean, SD, and n for each variable, grouped by upwelling class.
#' Also includes unclassified dates as a separate group.
#'
#' @param full_data Output from get_full_scenario_data()
#' @param include_unclassified If TRUE, includes dates with NA upwelling as "unclassified"
#' @return List with: summary (grouped stats), coverage (n per variable), metadata
summarize_full_scenario_detailed <- function(full_data, include_unclassified = TRUE) {
  
  # Optionally include unclassified dates
  data_for_summary <- full_data %>%
    mutate(upwelling_group = ifelse(is.na(upwelling), "unclassified", upwelling))
  
  if (!include_unclassified) {
    data_for_summary <- data_for_summary %>% filter(upwelling_group != "unclassified")
  }
  
  # --- Summary statistics by group ---
  summary_stats <- data_for_summary %>%
    group_by(upwelling_group) %>%
    summarize(
      n_dates = n(),
      
      # Depth
      depth_cutoff_mean = mean(depth_cutoff, na.rm = TRUE),
      
      # Phytoplankton
      n_phyto = sum(!is.na(micro_mmolN)),
      micro_mean = mean(micro_mmolN, na.rm = TRUE),
      micro_sd = sd(micro_mmolN, na.rm = TRUE),
      nano_mean = mean(nano_mmolN, na.rm = TRUE),
      nano_sd = sd(nano_mmolN, na.rm = TRUE),
      pico_mean = mean(pico_mmolN, na.rm = TRUE),
      pico_sd = sd(pico_mmolN, na.rm = TRUE),
      TotChlA_mean = mean(TotChlA_mmolN, na.rm = TRUE),
      TotChlA_sd = sd(TotChlA_mmolN, na.rm = TRUE),
      
      # Size fractions
      micro_frac_mean = mean(micro_frac, na.rm = TRUE),
      nano_frac_mean = mean(nano_frac, na.rm = TRUE),
      pico_frac_mean = mean(pico_frac, na.rm = TRUE),
      
      # Nutrients
      n_NO3 = sum(!is.na(NO3_mmolN)),
      NO3_mean = mean(NO3_mmolN, na.rm = TRUE),
      NO3_sd = sd(NO3_mmolN, na.rm = TRUE),
      
      n_PON = sum(!is.na(PON_mmolN)),
      PON_mean = mean(PON_mmolN, na.rm = TRUE),
      PON_sd = sd(PON_mmolN, na.rm = TRUE),
      
      # Zooplankton
      n_zoo = sum(!is.na(zoo_gt200_mmolN)),
      zoo_gt200_mean = mean(zoo_gt200_mmolN, na.rm = TRUE),
      zoo_gt200_sd = sd(zoo_gt200_mmolN, na.rm = TRUE),
      zoo_gt500_mean = mean(zoo_gt500_mmolN, na.rm = TRUE),
      zoo_gt500_sd = sd(zoo_gt500_mmolN, na.rm = TRUE),
      
      # Rates
      n_PP = sum(!is.na(PP_mgC_m2_d)),
      PP_mean = mean(PP_mgC_m2_d, na.rm = TRUE),
      PP_sd = sd(PP_mgC_m2_d, na.rm = TRUE),
      
      n_export = sum(!is.na(export_flux_mmolN)),
      export_mean = mean(export_flux_mmolN, na.rm = TRUE),
      export_sd = sd(export_flux_mmolN, na.rm = TRUE),
      
      # Environment
      Temp_mean = mean(Temp_C, na.rm = TRUE),
      Temp_sd = sd(Temp_C, na.rm = TRUE),
      Isotherm_21_mean = mean(Isotherm_21, na.rm = TRUE),
      
      .groups = "drop"
    )
  
  # --- Coverage table (long format, easier to read) ---
  coverage <- data_for_summary %>%
    group_by(upwelling_group) %>%
    summarize(
      total_dates = n(),
      phyto_size = sum(!is.na(micro_mmolN)),
      TotChlA = sum(!is.na(TotChlA_mmolN)),
      NO3 = sum(!is.na(NO3_mmolN)),
      PON = sum(!is.na(PON_mmolN)),
      PP = sum(!is.na(PP_mgC_m2_d)),
      Temperature = sum(!is.na(Temp_C)),
      zoo_gt200 = sum(!is.na(zoo_gt200_mmolN)),
      zoo_gt500 = sum(!is.na(zoo_gt500_mmolN)),
      export_flux = sum(!is.na(export_flux_mmolN)),
      Isotherm_21 = sum(!is.na(Isotherm_21)),
      .groups = "drop"
    )
  
  # --- Metadata ---
  metadata <- get_scenario_metadata()
  
  # --- Print summary ---
  cat("\n")
  cat("╔══════════════════════════════════════════════════════════════════╗\n")
  cat("║           CARIACO SCENARIO DATA SUMMARY                          ║\n")
  cat("╚══════════════════════════════════════════════════════════════════╝\n\n")
  
  cat("=== Data Coverage (n observations per variable) ===\n")
  print(as.data.frame(coverage), row.names = FALSE)
  
  cat("\n=== Integration Depth ===\n")
  cat(sprintf("  Mode: %s\n", attr(full_data, "depth_mode") %||% "not recorded"))
  summary_stats %>%
    select(upwelling_group, depth_cutoff_mean) %>%
    print(row.names = FALSE)
  
  cat("\n=== Chlorophyll a & Size Classes (mmol N m⁻³) ===\n")
  summary_stats %>%
    select(upwelling_group, n_phyto, 
           TotChlA_mean, micro_mean, nano_mean, pico_mean) %>%
    mutate(across(where(is.numeric) & !matches("^n_"), ~round(., 4))) %>%
    print(row.names = FALSE)
  
  cat("\n=== Size Fractions (dimensionless) ===\n")
  summary_stats %>%
    select(upwelling_group, micro_frac_mean, nano_frac_mean, pico_frac_mean) %>%
    mutate(across(where(is.numeric), ~round(., 3))) %>%
    print(row.names = FALSE)
  
  cat("\n=== Nutrients (mmol N m⁻³) ===\n")
  summary_stats %>%
    select(upwelling_group, n_NO3, NO3_mean, NO3_sd, n_PON, PON_mean, PON_sd) %>%
    mutate(across(where(is.numeric) & !matches("^n_"), ~round(., 3))) %>%
    print(row.names = FALSE)
  
  cat("\n=== Zooplankton Biomass (mmol N m⁻³) ===\n")
  cat("  NOTE: Net tows 0-200m, not matched to euphotic depth\n")
  summary_stats %>%
    select(upwelling_group, n_zoo, zoo_gt200_mean, zoo_gt200_sd, zoo_gt500_mean) %>%
    mutate(across(where(is.numeric) & !matches("^n_"), ~round(., 4))) %>%
    print(row.names = FALSE)
  
 cat("\n=== Primary Production (mg C m⁻² d⁻¹) ===\n")
  summary_stats %>%
    select(upwelling_group, n_PP, PP_mean, PP_sd) %>%
    mutate(across(where(is.numeric) & !matches("^n_"), ~round(., 1))) %>%
    print(row.names = FALSE)
  
  cat("\n=== Export Flux (mmol N m⁻² d⁻¹) ===\n")
  cat("  NOTE: Sediment trap at 225m, no Martin correction applied\n")
  summary_stats %>%
    select(upwelling_group, n_export, export_mean, export_sd) %>%
    mutate(across(where(is.numeric) & !matches("^n_"), ~round(., 4))) %>%
    print(row.names = FALSE)
  
  cat("\n=== Environment ===\n")
  summary_stats %>%
    select(upwelling_group, Temp_mean, Isotherm_21_mean) %>%
    mutate(across(where(is.numeric), ~round(., 2))) %>%
    print(row.names = FALSE)
  
  cat("\n=== Methodology Notes ===\n")
  cat("  • Upwelling classification: T(50m) ≤ 22°C = upwelling, > 22°C = relaxed\n")
  cat("  • Phytoplankton: HPLC Chl a → size fractions via diagnostic pigments\n")
  cat("  • Unit conversion: mg Chl × (C:Chl=50) / 12.01 × (N:C=16/106) → mmol N\n")
  cat("  • Zooplankton: mg DW × (C:DW=0.4) / 12.01 × (N:C=16/106) → mmol N\n")
  cat("  • PP: Niskin integrated, ×12 for 12h tropical daylight\n")
  cat("  • Export: 225m trap, duration-weighted monthly means\n")
  
  return(list(
    summary = summary_stats,
    coverage = coverage,
    metadata = metadata
  ))
}