library(tidyverse)
library(janitor)
library(reticulate)
library(countrycode)
library(Hmisc)
library(lubridate)
library(skimr)
library(haven)
library(scales)

# Creating different indices to measure covid_outcomes

df_covid_index <- df_all_yr %>%
  select(
    country_standard, 
    cum_total_cases_per_million, 
    cum_total_deaths_per_million,
    cum_total_tests_per_thousand) %>%
  filter(if_any(cum_total_cases_per_million:cum_total_deaths_per_million, ~ !is.na(.))) %>%  
  distinct() %>%
  mutate(
        # Use a scale of 1 - 10 to avoid having "0" in denominator
    stand_total_cases_per_million  = rescale(cum_total_cases_per_million, to = c(1, 10)),
        # Evan tested looking at logs of variables, like below
    # stand_total_cases_per_million  = rescale(-log(cum_total_cases_per_million), to = c(1, 10)),    
    stand_total_cases_per_million_rev  = rescale(-cum_total_cases_per_million, to = c(1, 10)),    
    stand_total_deaths_per_million = rescale(cum_total_deaths_per_million, to = c(1, 10)),
    stand_total_deaths_per_million_rev = rescale(-cum_total_deaths_per_million, to = c(1, 10)),
        # Lines below are for populating NA rows of deaths/million.
        # There are 11 countries with missing values. Doing this does not change the data very much
    # stand_total_deaths_per_million_no_NA = ifelse(
    #   is.na(stand_total_deaths_per_million), 
    #   mean(stand_total_deaths_per_million, na.rm = TRUE), 
    #   stand_total_deaths_per_million),    
    stand_total_tests_per_thousand = rescale(cum_total_tests_per_thousand, to = c(1, 10)),
    stand_total_tests_per_thousand_no_NA = ifelse(
      is.na(stand_total_tests_per_thousand), 
      1, 
      stand_total_tests_per_thousand),
    sum_stand_indicators = -stand_total_cases_per_million - stand_total_deaths_per_million + stand_total_tests_per_thousand_no_NA,
    numerator = stand_total_tests_per_thousand_no_NA,
    denominator_one = (stand_total_cases_per_million * stand_total_deaths_per_million),
    denominator_two = stand_total_cases_per_million + stand_total_deaths_per_million,
    covid_index1 = rescale(sum_stand_indicators),
    covid_index2 = rescale(numerator/denominator_one),
    covid_index3 = rescale(numerator/denominator_two),
    covid_index4 = rescale(
      (1/stand_total_cases_per_million) +
        (1/stand_total_deaths_per_million) +
        stand_total_tests_per_thousand_no_NA),
    covid_index5 = rescale(
      stand_total_tests_per_thousand_no_NA *
        ((1/stand_total_cases_per_million)+(1/stand_total_deaths_per_million))),
    covid_index6 = 
      rescale(stand_total_cases_per_million_rev * 
                stand_total_deaths_per_million_rev * 
                stand_total_tests_per_thousand_no_NA),
    # Line below can be used for examining natural log of indices
    # Evan looked at different results from log indices, 
    # but it didn't seem like a convincing approach
    #covid_index_log = covid_index %>% log() %>% rescale()
  )

histogram(df_covid_index$covid_index1)
histogram(df_covid_index$covid_index2)
histogram(df_covid_index$covid_index3)
histogram(df_covid_index$covid_index4)
histogram(df_covid_index$covid_index5)
histogram(df_covid_index$covid_index6)

df_i1 <- df_covid_index %>%
  arrange(desc(covid_index1)) %>%
  mutate(rank = 1:nrow(df_covid_index),
         country_1 = country_standard) %>%
  select(covid_index1, country_1, rank)

df_i2 <- df_covid_index %>%
  arrange(desc(covid_index2)) %>%
  mutate(rank = 1:nrow(df_covid_index),
         country_2 = country_standard) %>%
  select(covid_index2, country_2, rank)

df_i3 <- df_covid_index %>%
  arrange(desc(covid_index3)) %>%
  mutate(rank = 1:nrow(df_covid_index),
         country_3 = country_standard) %>%
  select(covid_index3, country_3, rank)

df_i4 <- df_covid_index %>%
  arrange(desc(covid_index4)) %>%
  mutate(rank = 1:nrow(df_covid_index),
         country_4 = country_standard) %>%
  select(covid_index4, country_4, rank)

df_i5 <- df_covid_index %>%
  arrange(desc(covid_index5)) %>%
  mutate(rank = 1:nrow(df_covid_index),
         country_5 = country_standard) %>%
  select(covid_index5, country_5, rank)

df_i6 <- df_covid_index %>%
  arrange(desc(covid_index6)) %>%
  mutate(rank = 1:nrow(df_covid_index),
         country_6 = country_standard) %>%
  select(covid_index6, country_6, rank)

df_i_all <- df_i1 %>% 
  full_join(df_i2) %>%
  full_join(df_i3) %>%
  full_join(df_i4) %>%
  full_join(df_i5) %>%
  full_join(df_i6) %>%
  select(rank, country_1, country_2, country_3, country_4, country_5, country_6, everything())

# Look at df_i_all to see how countries are ranking according to the different indices

# Running regressions of the control variables on the different indices
mod_test <- df_covid_index %>% 
  select(country_standard, covid_index1, covid_index2, covid_index3, covid_index4, covid_index5, covid_index6) %>%
  full_join(df_covid_controls, by = "country_standard")


df_mod_test <- mod_test %>% 
  pivot_longer(starts_with("covid_index"), names_to = "index", values_to = "index_score")

n_mod_test <- df_mod_test %>% select(-country_standard) %>% group_by(index) %>% nest()

mod_fun <- function(df)lm(index_score~., data = df)

m_mod_test <- n_mod_test %>% mutate(model=map(data, mod_fun))

m_mod_test[[3]][[1]] %>% summary()
m_mod_test[[3]][[2]] %>% summary()
m_mod_test[[3]][[3]] %>% summary()
m_mod_test[[3]][[4]] %>% summary()
m_mod_test[[3]][[5]] %>% summary()
m_mod_test[[3]][[6]] %>% summary()

# Index 1 has the highest F-Statistic, but a review of df_i_all shows that it is a bad index, 
  # based on how it ranks countries

# Index 4 seems to be the best pick