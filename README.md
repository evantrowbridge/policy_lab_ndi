# policy_lab_ndi

GitHub for Sharing NDI Code

## Notes on Data:

-   merged_data.csv

    -   A file that merges variables of interest from the data sets below.
    -   This file is large

-   merged_data_yr.csv

    -   Same as merged_data.csv but each row represents a year instead of an individual date

-   owid-covid-data.csv

    -   This comes from Our Wold in Data.
    -   Data can be updated to most recent version here: <https://github.com/owid/covid-19-data/tree/master/public/data>

-   EIU_democracy_index_clean.csv

    -   This comes from the [Economist Intelligence Unit (EIU) Democracy Index](https://www.eiu.com/n/campaigns/democracy-index-2020/).
    -   EIU published the data in a PDF. Gapminder collected and combined the data from the years 2006 to 2019 ([available here](https://www.gapminder.org/data/documentation/democracy-index/)). The 2020 pdf data was also published on the Democracy Index [Wikipedia page](https://en.wikipedia.org/wiki/Democracy_Index). We merge these data sets into a single data set.

-   public-health-expenditure-share-GDP-OWID.csv

    -   Public expenditure on health as a percent of GDP.
    -   Only covers years 2006 - 2014
    -   This comes from [Our World in Data](https://ourworldindata.org/grapher/public-health-expenditure-share-gdp-owid), using data from the WHO

-   OGP_data.xlsx

    -   Open Government Partnership Report data (2019)
        https://docs.google.com/spreadsheets/d/10NlwZZSGaJnRRZf0OD5ZsAmjYFDAhhUxXd88fMi32qY/edit#gid=261093125

-   GDP_data.xlsx

        World Bank GDP data
        https://data.worldbank.org/indicator/NY.GDP.MKTP.CD

-   CPI_Transparency_International.xlsx

    -   add info (the source and a link)?

-   Gini_coefficient_2020.csv

    -   Gives each country's Gini index for the year 2020

    -   add info (the source and a link)?

-   Africa_Integrity_Indictors_CLEAN_22021.xlsx

    -   Africa Integrity Indicators

        -   <https://www.africaintegrityindicators.org/data>

    -   Key Variables

        -   Transparency

            -   In practice,citizenscanaccessthe results and documents associated with procurement contracts (full contract, proposals, execution reports, financial audits, etc.).
            -   In law,citizens have a right to request public information from state bodies.
            -   In practice, citizen requests for public information are effective.
            -   In practice, citizens can access legislative processes and documents.
            -   In law, senior officials of the three branches of government (including heads of state and government, ministers, members of Parliament, judges, etc.) are required to disclose records of their assets and disclosures are public.
            -   In law, political parties are required to regularly disclose private donations.
            -   

        -   Corruption:

            -   In law, corruption is criminalized as a specific offense.
            -   In practice, the body/bodies that investigate/s allegations of public sector corruption is/are effective.
            -   In law, civil servants who report cases of corruption are protected from recrimination or other negative consequences.
            -   In law, there is an independent body/bodies mandated to receive and investigate cases of alleged public sector corruption.

-   WB_Public_Health_Expenditure_v2.csv

    -   World Health Organization Global Health Expenditure Database
    -   <https://data.worldbank.org/indicator/SH.XPD.GHED.PC.CD>

-   Worldwide_Governance_Indicators_clean.csv

    -   This comes from the [Worldwide_Governance_Indicators](https://info.worldbank.org/governance/wgi/).
    -   We merge data of two indicators (Voice and Accountability and Control of Corruption) into a single data set.

-   wgi_gov_effective.xlsx

    -   This comes from the [Worldwide_Governance_Indicators](https://info.worldbank.org/governance/wgi/).

    -   It includes the Government Effectiveness index scores

-   vdem_clean.csv

    -   This comes from the [V-Dem Dataset - Version 10](https://www.v-dem.net/en/data/data/v-dem-dataset/).
    -   Data can be updated to most recent version here: <https://www.v-dem.net/en/data/data/v-dem-dataset-v11/>
    -   We downloaded "Country-Year: V-Dem Core" data and merged three indicators (v2x_libdem: Liberal democracy index), v2x_partipdem: Participatory democracy index, v2x_cspart: Civil society participation index)
    -   This data goes back many years. We include data starting in 2006

-   PanDem_ts_V5.csv

    -   This data comes from V-Dem's ["Pandemic Backsliding" project](https://www.v-dem.net/en/our-work/research-projects/pandemic-backsliding/)

    -   The data was downloaded from [V-Dem's GitHub page](https://github.com/vdeminstitute/pandem)

-   Freedom_house.xlsx

    -   add info (the source and a link)?

-   wgi_gov_effective.xlsx

    -   "Government Effectiveness" combines into a single grouping responses on the quality of public service provision, the quality of the bureaucracy, the competence of civil servants, the independence of the civil service from political pressures, and the credibility of the government's commitment to policies. The main focus of this index is on "inputs" required for the government to be able to produce and implement good policies and deliver public goods.
    -   Years:1995-2018
    -   This comes from the [World Bank's Worldwide Governance Indicators (WGI)](http://info.worldbank.org/governance/wgi/).

-   
