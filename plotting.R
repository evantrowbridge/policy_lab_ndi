library(tidyverse)
library(dplyr)
library(ggplot2)
library(ggrepel)

setwd("C:/Users/Kimiko/OneDrive/??????????????????/5_winter 21/PPHA 60000 Policy Lab/final")

country_average_indices <- read.csv("country_average_indices.csv")
indices_and_controls_cross_section <- read.csv("indices_and_controls_cross_section.csv")
indices_and_controls <- read.csv("indices_and_controls.csv")


#plot b/ transparency and accountability index
ggplot(data = country_average_indices, aes(x= mean_transparency_index, y = mean_accountability_index, label = country_standard)) +
  geom_point() +
  geom_text_repel() +
  labs(title = "Relationship between transparency and accountability", x = "mean of transparency index", y = "mean of accountability index") +
  theme(plot.title = element_text(hjust = 0.5))

#plot b/ transparency and corruption index
ggplot(data = country_average_indices, aes(x= mean_transparency_index, y = mean_corruption_index, label = country_standard)) +
  geom_point() +
  geom_text_repel() +
  labs(title = "Relationship between transparency and corruption", x = "mean of transparency index", y = "mean of corruption index") +
  theme(plot.title = element_text(hjust = 0.5))

#plot b/ transparency and trust index
ggplot(data = country_average_indices, aes(x= mean_transparency_index, y = mean_trust_index, label = country_standard)) +
  geom_point() +
  geom_text_repel() +
  labs(title = "Relationship between transparency and trust", x = "mean of transparency index", y = "mean of trust index") +
  theme(plot.title = element_text(hjust = 0.5))

#plot b/ transparency and effectiveness index
ggplot(data = country_average_indices, aes(x= mean_transparency_index, y = mean_effectiveness_index, label = country_standard)) +
  geom_point() +
  geom_text_repel() +
  labs(title = "Relationship between transparency and effectiveness", x = "mean of transparency index", y = "mean of effectiveness index") +
  theme(plot.title = element_text(hjust = 0.5))

#plot b/ transparency and budget participation
ggplot(data = indices_and_controls_cross_section, aes(x= transparency_index_2019, y = budget_participation_index, label = country_standard)) +
  geom_point() +
  geom_text_repel() +
  labs(title = "Relationship between transparency and budget participations", x = "mean of transparency index", y = "mean of budget participation index") +
  theme(plot.title = element_text(hjust = 0.5))

#plot b/ transparency and budget transparency index
ggplot(data = indices_and_controls_cross_section, aes(x= transparency_index_2019, y = budget_transparency_index, label = country_standard)) +
  geom_point() +
  geom_text_repel() +
  labs(title = "Relationship between transparency and budget transparency", x = "mean of transparency index", y = "mean of budget transparency index") +
  theme(plot.title = element_text(hjust = 0.5))


#plot b/ transparency and covid outcomes
ggplot(data = indices_and_controls_cross_section, aes(x= transparency_index_2019, y = covid_index, label = country_standard)) +
  geom_point() +
  geom_text_repel() +
  labs(title = "Relationship between transparency and covid outcomes", x = "transparency index in 2019", y = "covid index") +
  theme(plot.title = element_text(hjust = 0.5))

#plot b/ transparency and pandemic violation
ggplot(data = indices_and_controls_cross_section, aes(x= transparency_index_2019, y = pandemic_dem_violation_index, label = country_standard)) +
  geom_point() +
  geom_text_repel() +
  labs(title = "Relationship between transparency and pandemic violation scores", x = "transparency index in 2019", y = "pandemic violation scores") +
  theme(plot.title = element_text(hjust = 0.5))

#plot b/ age controls and transpaerncy 
ggplot(data = indices_and_controls_cross_section, aes(x= transparency_index_2019, y = median_age, label = country_standard)) +
  geom_point() +
  geom_text_repel() +
  labs(title = "Relationship between transparency and median age", x = "transparency index in 2019", y = "median age in 2019 ") +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(data = indices_and_controls_cross_section, aes(x= transparency_index_2019, y = aged_65_older, label = country_standard)) +
  geom_point() +
  geom_text_repel() +
  labs(title = "Relationship between transparency and ratio of 65+ persons", x = "transparency index in 2019", y = "ratio of 65+ persons in 2019") +
  theme(plot.title = element_text(hjust = 0.5))

#plot b/ age controls and GDP

ggplot(data = indices_and_controls_cross_section, aes(x= gdp_percap_ppp_covid, y = median_age, label = country_standard)) +
  geom_point() +
  geom_text_repel() +
  labs(title = "Relationship between GDP per capita and median age", x = "GDP per capita", y = "mean of median age") +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(data = indices_and_controls_cross_section, aes(x= gdp_percap_ppp_covid, y = aged_65_older, label = country_standard)) +
  geom_point() +
  geom_text_repel() +
  labs(title = "Relationship between GDP per capita and ratio of 65+ persons", x = "GDP per capita in 2019", y = "ratio of 65+ persons in 2019") +
  theme(plot.title = element_text(hjust = 0.5))