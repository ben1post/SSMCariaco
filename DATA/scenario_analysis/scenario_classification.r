# =============================================================================
# SCENARIO CLASSIFICATION FUNCTIONS
# For CARIACO Basin upwelling regime analysis
# 
# This script provides functions to:
# 1. Load and process CTD data to derive isotherm depths, MLD, SST
# 2. Classify upwelling scenarios based on temperature at 50m
# 3. Load observed euphotic depth data (Pinckney)
# 4. Combine all scenario-relevant variables for analysis
#
# Usage (from Jupyter notebook in scenario_analysis/):
#   source("scenario_classification.R")
#   scenario_data <- get_combined_scenario_data()
# =============================================================================

library(tidyverse)
library(oce)

# Source interpolation functions (one level up)
source("../interpolateData.r")

# =============================================================================
# 1. CTD-DERIVED SCENARIO VARIABLES
# =============================================================================

#' Load and process CTD data for scenario classification
#'
#' Processes raw CTD data to extract:
#' - 21°C isotherm depth (proxy for thermocline/upwelling)
#' - Mixed layer depth (MLD) from sigma_t criterion
#' - Sea surface temperature (SST, mean 0-10m)
#' - Upwelling index based on temperature at 50m
#'
#' @return A data frame with columns: date, Isotherm_21, MLD, sst, temp_50m, ui, upwelling
load_ctd_scenario_data <- function() {


  # --- Load raw CTD data ---
  ctd_ds <- read.csv("../BCO-DMO/ctd.csv", na.strings = "nd")
  
  # Parse date

  ctd_ds$date <- paste(ctd_ds$Year, '-', ctd_ds$Month, '-', ctd_ds$Day, sep = '')
  ctd_ds$date <- as.Date(ctd_ds$date, format = "%Y-%m-%d")
  
  # --- Interpolate temperature ---
  ctd_temp_int <- interpolateDF(prepdataframe(ctd_ds, "temp"))
  
  # --- 21°C Isotherm Depth ---
  iso21_df <- ctd_temp_int %>%
    group_by(date) %>%
    filter(depth > 6) %>%
    mutate(iso21 = value_int < 21) %>%
    filter(iso21 == TRUE) %>%
    slice(1) %>%
    rename(Isotherm_21 = depth) %>%
    select(date, Isotherm_21) %>%
    ungroup()
  
  # --- Mixed Layer Depth (sigma_t criterion) ---
  ctd_sigma_t_int <- interpolateDF(prepdataframe(ctd_ds, "sigma_t"))
  
  mld_df <- ctd_sigma_t_int %>%
    group_by(date) %>%
    filter(depth > 9) %>%
    # Criterion: sigma_t deviates by >= 0.2 from surface value
    mutate(mld = value_int >= first(value_int) + 0.2 | 
             value_int <= first(value_int) - 0.2) %>%
    filter(mld == TRUE) %>%
    slice(1) %>%
    rename(MLD = depth) %>%
    select(date, MLD) %>%
    ungroup()
  

  # --- SST (mean 0-10m) ---
  sst_df <- ctd_temp_int %>%
    group_by(date) %>%
    filter(depth <= 10) %>%
    summarize(sst = mean(value_int, na.rm = TRUE), .groups = "drop")
  
  # --- Upwelling Index (temperature at 50m) ---
  upwelling_df <- ctd_temp_int %>%
    group_by(date) %>%
    filter(depth == 50) %>%
    mutate(
      temp_50m = value_int,
      ui = case_when(
        value_int <= 20.0 ~ "strong",
        value_int <= 21.0 ~ "moderate",
        value_int <= 22.0 ~ "weak",
        value_int > 22.0  ~ "relaxed"
      ),
      upwelling = case_when(
        value_int <= 22.0 ~ "upwelling",
        value_int > 22.0  ~ "relaxed"
      )
    ) %>%
    select(date, temp_50m, ui, upwelling) %>%
    ungroup()
  
  # --- Combine all CTD-derived variables ---
  ctd_scenario_data <- list(iso21_df, mld_df, sst_df, upwelling_df) %>%
    reduce(full_join, by = "date") %>%
    arrange(date)
  
  return(ctd_scenario_data)
}

# =============================================================================
# 2. OBSERVED EUPHOTIC DEPTH DATA
# =============================================================================
#' Load observed euphotic depth data (Pinckney)
#'
#' Loads the MLD2EuZ dataset containing measured euphotic depths
#' (1% light level) and isotherm depths from direct measurements.
#'
#' @return A data frame with columns: date, euphotic_depth_obs, plus original columns
load_euphotic_depth_data <- function() {
  
  euz_ds <- read.csv("../processed/MLD2EuZ_2.csv")
  
  # Parse date
  euz_ds$date <- as.Date(euz_ds$Date, format = "%Y-%m-%d")
  
  # Rename euphotic depth column for clarity
  euz_clean <- euz_ds %>%
    rename(euphotic_depth_obs = x1) %>%
    select(date, euphotic_depth_obs, X21degC, X22degC, MLD2015, MLD2019)
  
  return(euz_clean)
}

# =============================================================================
# 3. COMBINED SCENARIO DATA
# =============================================================================

#' Get combined scenario classification data
#'
#' Merges CTD-derived variables with observed euphotic depth data.
#' This is the main function to call for scenario analysis.
#'
#' @return A data frame with all scenario-relevant variables by date
get_combined_scenario_data <- function() {
  
  ctd_data <- load_ctd_scenario_data()
  euz_data <- load_euphotic_depth_data()
  
  combined <- full_join(ctd_data, euz_data, by = "date") %>%
    arrange(date) %>%
    # Add year/month for aggregation
    mutate(
      year = format(date, "%Y"),
      month = format(date, "%m"),
      year_month = format(date, "%Y-%m")
    )
  
  return(combined)
}

# =============================================================================
# 4. SUMMARY HELPER FUNCTIONS
# =============================================================================

#' Summarize scenario data by upwelling class
#'
#' @param scenario_data Output from get_combined_scenario_data()
#' @return Summary statistics grouped by upwelling class
summarize_by_upwelling <- function(scenario_data) {
  
  scenario_data %>%
    filter(!is.na(upwelling)) %>%
    group_by(upwelling) %>%
    summarize(
      n = n(),
      mean_iso21 = mean(Isotherm_21, na.rm = TRUE),
      sd_iso21 = sd(Isotherm_21, na.rm = TRUE),
      mean_MLD = mean(MLD, na.rm = TRUE),
      sd_MLD = sd(MLD, na.rm = TRUE),
      mean_sst = mean(sst, na.rm = TRUE),
      mean_temp_50m = mean(temp_50m, na.rm = TRUE),
      mean_euphotic_obs = mean(euphotic_depth_obs, na.rm = TRUE),
      sd_euphotic_obs = sd(euphotic_depth_obs, na.rm = TRUE),
      n_euphotic_obs = sum(!is.na(euphotic_depth_obs)),
      .groups = "drop"
    )
}

#' Summarize scenario data by detailed upwelling index
#'
#' @param scenario_data Output from get_combined_scenario_data()
#' @return Summary statistics grouped by ui (strong/moderate/weak/relaxed)
summarize_by_ui <- function(scenario_data) {
  
  scenario_data %>%
    filter(!is.na(ui)) %>%
    group_by(ui) %>%
    summarize(
      n = n(),
      mean_iso21 = mean(Isotherm_21, na.rm = TRUE),
      sd_iso21 = sd(Isotherm_21, na.rm = TRUE),
      mean_euphotic_obs = mean(euphotic_depth_obs, na.rm = TRUE),
      sd_euphotic_obs = sd(euphotic_depth_obs, na.rm = TRUE),
      n_euphotic_obs = sum(!is.na(euphotic_depth_obs)),
      .groups = "drop"
    ) %>%
    # Order factor levels logically
    mutate(ui = factor(ui, levels = c("strong", "moderate", "weak", "relaxed"))) %>%
    arrange(ui)
}