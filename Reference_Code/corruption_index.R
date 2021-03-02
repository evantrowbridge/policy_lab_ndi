library(tidyverse)
library(dplyr)
library(ggplot2)

data <- read.csv("data/merged_data_yr.csv")


#transform wgi from factor to numeric
data2 <- data
data2$wgi_voice_and_accountability <- as.character(data$wgi_voice_and_accountability)
data2$wgi_voice_and_accountability <- as.numeric(data2$wgi_voice_and_accountability)
data2$wgi_control_of_corruption <- as.character(data$wgi_control_of_corruption)
data2$wgi_control_of_corruption <- as.numeric(data2$wgi_control_of_corruption)
class(data2$wgi_control_of_corruption)

#checking each variable
summary(data2$cpi_score)
summary(data2$wgi_control_of_corruption)

#rescaling and creating index of corruption (cpi_score: 0-100, WGI: -2.5~2.5)
corruption_index_pre <- data2 %>%
  mutate(data2, cpi_score_scale = data2$cpi_score/100) %>%
  mutate(data2, wgi_coc_scale = (data2$wgi_control_of_corruption+2.5)/5) %>%
  mutate(data2, corruption_idx_sum = ifelse(year >= 2012, cpi_score_scale + wgi_coc_scale, NA )) %>%
  mutate(data2, corruption_idx_avg = ifelse(year >= 2012, (cpi_score_scale + wgi_coc_scale)/2, NA ))

#checking rescaled variable and index
summary(corruption_index_pre$cpi_score)
summary(corruption_index_pre$wgi_control_of_corruption)
summary(corruption_index_pre$cpi_score_scale)
summary(corruption_index_pre$wgi_coc_scale)

#create csv file of index
write.csv(corruption_index_pre[c(1:2, 109:110)], "Corruption index.csv")

#plotting rescaled variables
avg_cpi_score_scale <- corruption_index_pre %>%
  group_by(year) %>%
  summarise(mean_cpi_scale = mean(cpi_score_scale, na.rm = TRUE))

ggplot(data = avg_cpi_score_scale, aes(x=year, y =mean_cpi_scale)) +
geom_line() + labs(title = "Average of rescaled Perceptions of corruption", y = "Average of rescaled cpi") +
theme(plot.title = element_text(hjust = 0.5))


avg_wgicoc_score_scale <- corruption_index_pre %>%
  group_by(year) %>%
  summarise(mean_wgicoc_scale = mean(wgi_coc_scale, na.rm = TRUE))

ggplot(data = avg_wgicoc_score_scale, aes(x=year, y =mean_wgicoc_scale)) +
  geom_line() + labs(title = "Average of rescaled WGI CoC", y = "Average of rescaled WGI CoC") +
  theme(plot.title = element_text(hjust = 0.5))


ggplot(data = corruption_index_pre, aes(x= cpi_score_scale, y = wgi_coc_scale)) +
  geom_point() + labs(title = "Perceptions of corruption and WGI CoC", x = "rescaled wgi_Coc", y = "rescaled cpi") +
  theme(plot.title = element_text(hjust = 0.5))


#cheking index
summary(corruption_index_pre$corruption_idx_sum)
summary(corruption_index_pre$corruption_idx_avg)

#plotting index
avg_corruption_idx_sum <- corruption_index_pre %>%
  group_by(year) %>%
  summarise(mean_corruption_idx_sum = mean(corruption_idx_sum, na.rm = TRUE))

ggplot(data = avg_corruption_idx_sum, aes(x=year, y =mean_corruption_idx_sum)) +
  geom_line() + labs(title = "Average of Corruption index (sum)", y = "Average of Corruption index (sum)") +
  theme(plot.title = element_text(hjust = 0.5))

avg_corruption_idx_avg <- corruption_index_pre %>%
  group_by(year) %>%
  summarise(mean_corruption_idx_avg = mean(corruption_idx_avg, na.rm = TRUE))

ggplot(data = avg_corruption_idx_avg, aes(x=year, y =mean_corruption_idx_avg)) +
  geom_line() + labs(title = "Average of Corruption index (avg)", y = "Average of Corruption index (avg)") +
  theme(plot.title = element_text(hjust = 0.5))
