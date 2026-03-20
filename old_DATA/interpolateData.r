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
        
    VarDF <- ds %>%
      select(date, all_of(variable), depth) %>%
      gather(key='key',value = "value", -date, -depth)

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

#' Takes data frame created with "prepdataframe" function and interpolates along depth
#'
#' The function interpolates the given data according to the provided algorithm,
#' which can be either linear or using the "unesco" interpolation function provided
#' by the "oce" R package. Previous tests with the "rr" variant resulted in some out 
#' of bound (negative) values resulting from the interpolation.
#'
#' @param DF A data frame
#' @param func A character vector to choose the interpolation algorithm to use
#' @returns An interpolated data frame from 0 to 200 meters, using all available depth intervals
interpolateDF <- function(DF, func='unesco'){
    zz <- seq(0, 200, 1)
    
    if(func=='linear'){
        IntDF <- DF %>%
            group_by(date) %>%
            filter(sum(!is.na(value))>1) %>%
            do(data.frame(value_int = with(.,approx(depth, value, zz)), depth = zz)) 
        
        IntDF <- IntDF %>% 
              rename(
                value_int = value_int.y
                )
        IntDF$value_int.x <- NULL
        }
    
    else if(func=='unesco'){
        IntDF <- DF %>%
            group_by(date) %>%
            do(data.frame(value_int = with(.,oceApprox(depth, value, zz, "unesco")), depth = zz)) 
        }

    #checkInterpolation(IntDF, DF)
    
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
#' @returns A data frame with just the variable, date and depth
interpolateData <- function(ds, var, int_func='unesco', depth_to=100, depth_from=0, noofNA=40){
 
    ds_dat <- prepdataframe(ds, var)

    ds_int <- interpolateDF(ds_dat, int_func)   

    #print(ds_int %>% slice_min(value_int))
    
    ds_sum <- ds_int %>%
        group_by(date) %>%
        filter(depth_from<=depth & depth<=depth_to) %>%
        filter(sum(is.na(value_int))<noofNA) %>%
        summarize(value_sum = sum(value_int, na.rm=TRUE), 
                  var = mean(value_int, na.rm=TRUE),
                  NAs = sum(is.na(value_int)), .groups="keep")
    # is this necessary?
    ds_sum_monthly <- ds_sum %>%
          mutate(time_month = format(date, format="%m-%Y"))
    
    ds_return <- data.frame(dat_var = ds_sum_monthly$var, date = ds_sum_monthly$date)
    
    return(ds_return)
}