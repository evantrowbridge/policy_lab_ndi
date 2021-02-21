library(tidyverse)
library(lubridate)

# Importing Years 2006 to 2019. 
# Original source: https://www.gapminder.org/data/documentation/democracy-index/
eiu_dem_idx_2006_2019_import <- read_csv("data/raw_data/EIU-Democracy Indices - data-for-countries-etc-by-year.csv") %>%
  rename(Country = name)

# Importing 2020 Data
# Source: https://en.wikipedia.org/wiki/Democracy_Index
eiu_dem_idx_2020_import <- read_csv("data/raw_data/eiu-Democracy Index-2020.csv")
eiu_dem_idx_2020_import <- eiu_dem_idx_2020_import %>% 
  mutate(across(Electoral:Overall_score,  ~. * 10)) %>%
  mutate(
    Score_Change = NA_integer_,
    time = 2020)

df_DI <- bind_rows(eiu_dem_idx_2006_2019_import, eiu_dem_idx_2020_import)

df_DI <- df_DI %>%
  mutate(Date = ymd(time, truncated = 2L),
         Year = year(Date)) %>%
  select(-time) %>%
  arrange(Country, -Year) 

# The score change from Wikipedia is absolute values so I'm just going to calculate manually
df_DI <- df_DI %>%
  mutate(
    Score_Change_new = ifelse(
      Year == 2020,
      round(Overall_score - lead(Overall_score), 2),
      NA_integer_),
    Score_Change = ifelse(
      Year == 2020,
      Score_Change_new,
      Score_Change)) %>%
  select(-Score_Change_new)

df_DI <- df_DI %>%
  mutate(post_covid = 
           ifelse(Year >= 2020,
                  TRUE,
                  FALSE))

write_csv(df_DI, "data/EIU_democracy_index_clean.csv")
