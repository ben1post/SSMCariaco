# This script provides functions to prepare input data and interpolate the CARIACO Time-Series data along depth
# and summarize the counts or concentrations.

library(oce)
library(tidyverse)

#' Extract variable from dataframe and rename for interpolation
#'
#' That is how it works
#' some more docs here.
#'
#' @param ds A data frame
#' @param variable A character vector that is a column name in the dataframe to be extracted, together with depth and date
#' @returns A data frame with just the variable, date and depth
prepdataframe <- function(ds, variable=''){
  
  # Ensure variable is passed as a string/character
  var_name <- as.character(variable)
  
  VarDF <- ds %>%
    select(date, all_of(var_name), depth) %>%
    # Use pivot_longer instead of gather
    pivot_longer(
      cols = all_of(var_name), 
      names_to = "key", 
      values_to = "value"
    )
  
  return(VarDF)
}


#' Diagnostic function that checks interpolated values for outliers and non-sensical data points
#'
#' The function takes the interpolated data frame and checks if there are negative values or outliers
#' and gives a warning and plots the depth transects.
#'
#' @param intDF A data frame of interpolated data
#' @param DF A data frame of original uninterpolated data
#' @param func A character vector to choose the interpolation algorithm to use
#' @returns An interpolated data frame from 0 to 200 meters, using all available depth intervals
checkInterpolation <- function(intDF,DF){
  check.max <- function(x) ifelse( !all(is.na(x)), max(x, na.rm=T), NA)
  check.min <- function(x) ifelse( !all(is.na(x)), min(x, na.rm=T), NA)
  
  diagnostic_intDF <- intDF %>% group_by(date) %>% summarize(NAs=sum(is.na(value_int)), 
                                                             min=check.min(value_int),
                                                             max=check.max(value_int))
  diagnostic_DF <- DF %>% group_by(date) %>% summarize(NAs=sum(is.na(value)), 
                                                       min=check.min(value),
                                                       max=check.max(value))
  diagDF <- left_join(diagnostic_DF, diagnostic_intDF, by="date", suffix = c(".og", ".int"))
  
  #print(diagDF)
  
  dates <- unique(intDF$date)
  for(datx in dates[180:215]){
    intDF_subset <- intDF[intDF$date==datx,]
    DF_subset <- DF[DF$date==datx & DF$depth<=200,]
    print(plot(intDF_subset$value_int, -intDF_subset$depth),
          points(DF_subset$value, -DF_subset$depth, col="red"))
  }
}

#' #' Takes data frame created with "prepdataframe" function and interpolates along depth
#' #'
#' #' The function interpolates the given data according to the provided algorithm,
#' #' which can be either linear or using the "unesco" interpolation function provided
#' #' by the "oce" R package. Previous tests with the "rr" variant resulted in some out 
#' #' of bound (negative) values resulting from the interpolation.
#' #'
#' #' @param DF A data frame
#' #' @param func A character vector to choose the interpolation algorithm to use
#' #' @returns An interpolated data frame from 0 to 200 meters, using all available depth intervals
#' interpolateDF <- function(DF, func='unesco'){
#'   zz <- seq(0, 200, 1)
#'   
#'   if(func=='linear'){
#'     IntDF <- DF %>%
#'       group_by(date) %>%
#'       filter(sum(!is.na(value))>1) %>%
#'       do(data.frame(value_int = with(.,approx(depth, value, zz)), depth = zz)) 
#'     
#'     IntDF <- IntDF %>% 
#'       rename(
#'         value_int = value_int.y
#'       )
#'     IntDF$value_int.x <- NULL
#'   }
#'   
#'   else if(func=='unesco'){
#'     IntDF <- DF %>%
#'       group_by(date) %>%
#'       do(data.frame(value_int = with(.,oceApprox(depth, value, zz, "unesco")), depth = zz)) 
#'   }
#'   
#'   #checkInterpolation(IntDF, DF)
#'   
#'   return(IntDF)
#' }





#' Takes data frame created with "prepdataframe" function and interpolates along depth
#'
#' UPDATED: 
#' 1. Sorts data by depth.
#' 2. Checks if shallowest valid data point is <= 50m.
#' 3. If yes, duplicates that value at 0m (Surface) to anchor the interpolation.
#' 4. Runs the selected interpolation algorithm.
#'
#' @param DF A data frame
#' @param func A character vector: 'unesco' (spline) or 'linear'
#' @returns An interpolated data frame from 0 to 200 meters
interpolateDF <- function(DF, func='unesco', surface_fix=FALSE){
  zz <- seq(0, 200, 1)
  
  IntDF <- DF %>%
    group_by(date) %>%
    filter(sum(!is.na(value)) > 0) %>% # Remove dates with no data
    arrange(depth) %>%                 # Ensure sorted by depth
    do({
      # Extract valid data vectors
      d <- .$depth[!is.na(.$value)]
      v <- .$value[!is.na(.$value)]
      
      # --- STEP 1: SURFACE FIX ---
      # If the shallowest data is >0m but <=50m (i.e. 35m and above), add a 0m anchor point
      if(surface_fix && length(d) > 0 && d[1] > 0 && d[1] <= 50){
        d <- c(0, d)
        v <- c(v[1], v)
      }
      
      # --- STEP 2: INTERPOLATE ---
      if(func == 'linear'){
        # Linear needs 2 points. Rule=1 prevents extrapolation into deep NAs
        if(length(d) < 2) val_out <- rep(NA, length(zz))
        else val_out <- approx(d, v, zz, rule=1)$y
        
      } else {
        # Unesco (Reiniger-Ross Spline)
        val_out <- oceApprox(d, v, zz, "unesco")
      }
      
      data.frame(value_int = val_out, depth = zz)
    })
  
  return(IntDF)
}


#' Function to prepare, interpolate and calculate summary statistic of depth interpolated variable
#'
#' Wrapper function that uses "prepdataframe" and "interpolateDF" to prepare and interpolate a variable from
#' a data frame input. It allows choosing the interpolation algorithm and the depth interval for which to
#' calculate the mean value, as well as check for the number of NA values within the depth interval (threshold
#' to ignore mean with to many NAs given by noofNA parameter).
#'
#' @param ds A data frame
#' @param var A character vector that is a column name in the dataframe to be extracted and interpolated
#' @param int_func A character vector to choose interpolation algorithm, options: 'linear' and 'unesco'
#' @param output_type A character vector to choose output type: 'mean' for depth-averaged, 'integrated' for depth-integrated
#' @returns A data frame with just the variable, date and depth
interpolateData <- function(ds, var, int_func='unesco', depth_to=100, depth_from=0, noofNA=25, output_type='mean', surface_fix=FALSE){
  
  # 1. Prepare Data
  ds_dat <- prepdataframe(ds, var)
  
  # 2. Interpolate (Includes the new Surface Fix logic)
  ds_int <- interpolateDF(ds_dat, int_func, surface_fix=surface_fix)   
  
  # 3. Filter and Summarize
  ds_sum <- ds_int %>%
    group_by(date) %>%
    # Filter to specific depth layer (e.g. 0-100m)
    filter(depth >= depth_from & depth <= depth_to) %>%
    # Check NA Threshold: If too many NAs, the whole date is dropped here
    filter(sum(is.na(value_int)) < noofNA) %>%
    # Calculate stats
    summarize(
      value_sum = sum(value_int, na.rm=TRUE), 
      value_mean = mean(value_int, na.rm=TRUE),
      NAs = sum(is.na(value_int)), 
      .groups="drop" # ungroup immediately
    )
  
  # 4. Format Output based on type
  if(output_type == 'integrated'){
    ds_return <- data.frame(
      dat_var = ds_sum$value_mean * (depth_to - depth_from), 
      date = ds_sum$date
    )
  } else {
    ds_return <- data.frame(
      dat_var = ds_sum$value_mean, 
      date = ds_sum$date
    )
  }
  
  return(ds_return)
}

