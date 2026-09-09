# Purpose of the Script:
# This R script retrieves MODIS (Moderate Resolution Imaging Spectroradiometer) satellite data for Leaf Area Index (LAI)
# and its associated bands (Standard Deviation and Quality Control)/
# The script prepares it for batch processing, and then downloads the MODIS data for a specified period
# and spatial extent using the MODISTools package. The output is saved in a specified directory for further analysis.

#This code is the modification of the code from https://github.com/aukkola/PLUMBER2/blob/master/Get_MODIS_LAI_for_all_sites.R

# Load necessary libraries
library(MODISTools)  # MODISTools for accessing and processing MODIS satellite data
library(ncdf4)       # ncdf4 for working with netCDF format data files
library(readxl)      # readxl for reading Excel files, if needed

# Clear the R environment of all existing variables
rm(list = ls(all = TRUE))
#
# Load data from CSV file containing station information
data = read.csv("/Users/prajzwal/Downloads/01_stationinfo_L0_ICOS2025_with_heights.csv")

# Filter data to include only the selected stations: "NL-Loo", "NL-Vee", and "NL-Cab"
selected_stations = c("NL-Gla", "NL-Wen")  # Define selected station names
filtered_data = data[data$station_name %in% selected_stations, ]  # Subset data for selected stations

# Create a new DataFrame 'sites_to_fetch' with columns station_name, latitude, and longitude
sites_to_fetch = filtered_data[c('station_name', 'latitude', 'longitude')]
colnames(sites_to_fetch) = c('site_name', 'lat', 'lon')  # Rename columns for compatibility with MODISTools

# Define output directory for saving MODIS data
outdir <- "/Users/prajzwal/code/MODIS_Raw/"
# dir.create(outdir, recursive=TRUE) # Uncomment to create the directory if it does not exist

# Specify the MODIS product and bands to retrieve
product <- "MCD15A2H"  # MODIS/Terra+Aqua Leaf Area Index/FPAR 8-Day L4

# Define the specific bands of interest
band_lai <- "Lai_500m"        # Leaf Area Index band
band_sd <- "LaiStdDev_500m"   # Leaf Area Index standard deviation band
band_qc <- "FparLai_QC"       # Quality control band for LAI

# Define the radius around each site to fetch data in kilometers
km <- 1  # 1 km radius; note that using a radius smaller than this may cause issues

# Print status message indicating the start of batch processing for LAI
print("Batch processing LAI")

# Retrieve Leaf Area Index (LAI) data for the selected sites in batch mode
mt_batch_subset(
  df = sites_to_fetch,           # DataFrame containing site names, latitudes, and longitudes
  product = product,             # MODIS product to retrieve
  band = band_lai,               # Specific band to retrieve
  start = "2000-01-01",          # Start date for data retrieval
  end = format(Sys.time(), "%Y-%m-%d"),  # End date for data retrieval (current date)
  km_lr = km,                    # Number of km to extend from the center in longitude
  km_ab = km,                    # Number of km to extend from the center in latitude
  out_dir = outdir,              # Output directory for saving the data
  internal = FALSE               # Whether to keep the data internally or write to files
)

# Print status message indicating the start of batch processing for LAI SD
print("Batch processing SD")

# Retrieve Leaf Area Index standard deviation (SD) data
mt_batch_subset(
  df = sites_to_fetch,
  product = product,
  band = band_sd,
  start = "2000-01-01",
  end = format(Sys.time(), "%Y-%m-%d"),
  km_lr = km,
  km_ab = km,
  out_dir = outdir,
  internal = FALSE
)

# Print status message indicating the start of batch processing for QC
print("Batch processing QC")

# Retrieve Leaf Area Index Quality Control (QC) data
mt_batch_subset(
  df = sites_to_fetch,
  product = product,
  band = band_qc,
  start = "2000-01-01",
  end = format(Sys.time(), "%Y-%m-%d"),
  km_lr = km,
  km_ab = km,
  out_dir = outdir,
  internal = FALSE
)

# Print final status message indicating that batch processing is complete
print("Batch processing finished")


# # Load necessary libraries
# library(MODISTools)
# library(doParallel)
# library(foreach)
# library(parallel)
# 
# # Clear environment
# rm(list = ls(all = TRUE))
# 
# # Load and prepare data
# data = read.csv("/Users/prajzwal/Downloads/01_stationinfo_L0_ICOS2025_with_heights.csv")
# sites_to_fetch = data[c('station_name', 'latitude', 'longitude')]
# colnames(sites_to_fetch) = c('site_name', 'lat', 'lon')
# # Filter only for station_name in NL-Wen and NL-Gla
# sites_to_fetch <- sites_to_fetch[sites_to_fetch$site_name %in% c("NL-Wen", "NL-Gla"), ]
# 
# # Setup parallel processing
# num_cores <- detectCores() - 1  # Leave one core free
# registerDoParallel(cores = num_cores)
# 
# print(paste("Using", num_cores, "cores for parallel processing"))
# 
# # Define output directory
# outdir <- "/Users/prajzwal/code/MODIS_Raw/parallel"
# # dir.create(outdir, recursive = TRUE)
# 
# # Parallel processing function for individual sites
# process_site_parallel <- function(site_row) {
#   tryCatch({
#     # Download all three bands for this site at once
#     mt_subset(
#       product = "MCD15A2H",
#       band = c("Lai_500m", "LaiStdDev_500m", "FparLai_QC"),  # All bands together
#       lat = site_row$lat,
#       lon = site_row$lon,
#       start = "2000-01-01",
#       end = format(Sys.time(), "%Y-%m-%d"),
#       km_lr = 1,
#       km_ab = 1,
#       site_name = site_row$site_name,
#       out_dir = outdir,
#       internal = FALSE,
#       progress = FALSE
#     )
#     return(paste("Completed:", site_row$site_name))
#   }, error = function(e) {
#     return(paste("Error for", site_row$site_name, ":", e$message))
#   })
# }
# 
# # Execute parallel processing
# start_time <- Sys.time()
# results <- foreach(i = 1:nrow(sites_to_fetch), 
#                    .packages = c("MODISTools"),
#                    .combine = c) %dopar% {
#                      process_site_parallel(sites_to_fetch[i, ])
#                    }
# end_time <- Sys.time()
# 
# # Stop cluster
# stopImplicitCluster()
# 
# print(paste("Total processing time:", round(end_time - start_time, 2), "minutes"))
# print("Processing completed for all sites")
