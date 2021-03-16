# Harris Policy Lab Spring 2021: NDI Team

## Notes on Code:

The script for cleaning and organizing data is: "data_preparation.Rmd"

The script for running regressions is: "NDI Regressions.ipynb"

The script for creating plots is: "plotting.R"

## Notes on Data:

The panel data of the indices is located in the "data" folder, in a file named indices_and_controls.csv

The cross-sectional data (for Covid-19 related analysis) is located in the "data" folder, in a file named indices_and_controls_cross_section.csv

### Data used to create indices and perform analysis:

The files listed below are located in the Data folder

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

    -   Open Government Partnership Report data (2019) <https://docs.google.com/spreadsheets/d/10NlwZZSGaJnRRZf0OD5ZsAmjYFDAhhUxXd88fMi32qY/edit#gid=261093125>

-   CPI_Transparency_International_rev.xlsx

    -   This comes from Trensparency International's [Corruption Perceptions Index](https://www.transparency.org/en/cpi/2020/index/nzl).
    -   We multiplied each score between 2006 and 2012 by 10 to make it in 0-100 scale as same as after 2013.

-   Worldwide_Governance_Indicators_clean.csv

    -   This comes from the [Worldwide_Governance_Indicators](https://info.worldbank.org/governance/wgi/).
    -   We merge data of two indicators (Voice and Accountability and Control of Corruption) into a single data set.

-   wgi_gov_effective.xlsx

    -   "Government Effectiveness" combines into a single grouping responses on the quality of public service provision, the quality of the bureaucracy, the competence of civil servants, the independence of the civil service from political pressures, and the credibility of the government's commitment to policies. The main focus of this index is on "inputs" required for the government to be able to produce and implement good policies and deliver public goods.
    -   Years: 1995-2018
    -   This comes from the [World Bank's Worldwide Governance Indicators (WGI)](http://info.worldbank.org/governance/wgi/).

-   Vdem_account_transp_corrupt.csv

    - This data comes from the V-Dem data set
    - Variables
    1. transparent_laws: "Are the laws of the land clear, well publicized, coherent (consistent with each other), relatively stable from year to year, and enforced in a predictable manner?" (Ordinal, 0-10)
    2. accountability_index: "To what extent is the ideal of government accountability achieved?"
    - "Government accountability is understood as constraints on the government’s use of political power through requirements for     justification for its actions and potential sanctions. We organize the sub-types of accountability spatially. "
    
    3. vertical_index: "To what extent is the ideal of vertical government accountability achieved?"
    - " Vertical accountability captures the extent to which citizens have the power to hold the government accountable. The mechanisms of vertical accountability include formal political participation on part of the citizens — such as being able to freely organize in political parties — and participate in free and fair elections, including for the chief executive."
    
    4. diagonal_index: "To what extent is the ideal of diagonal government accountability achieved?"
    - "Diagonal accountability covers the range of actions and mechanisms that citizens, civil society organizations CSOs, and an independent media can use to hold the government accountable. These mechanisms include using informal tools such as social mobilization and investigative journalism to enhance vertical and horizontal accountability."
    
    5. horizontal_index - "To what extent is the ideal of horizontal government accountability achieved?"
    - "Horizontal accountability concerns the power of state institutions to oversee the government by demanding information, questioning officials and punishing improper behavior. This form of accountability ensures checks between institutions and prevents the abuse of power. The key agents in horizontal government accountability are: the legislature; the judiciary; and specific oversight agencies such as ombudsmen, prosecutor and comptroller generals."
    
  
    6. corruption_index: Political Corruption Index, "How pervasive is political corruption?"
    "The directionality of the V-Dem corruption index runs from less corrupt to more corrupt unlike the other V-Dem variables that generally run from less democratic to more democratic situation."
    
    7. exec_corruption: Executive Corruption Index, "How routinely do members of the executive, or their agents grant favors in exchange for bribes, kickbacks, or other material inducements, and how often do they steal, embezzle, or misappropriate public funds or other state resources for personal or family use?"
    
    8. public_sector_corruption: Public Sector Corruption Index, "To what extent do public sector employees grant favors in exchange for bribes, kickbacks, or other material inducements, and how often do they steal, embezzle, or misappropriate public funds or other state resources for personal or family use?"

    
    

-   PanDem_ts_V5.csv

    -   This data comes from V-Dem's ["Pandemic Backsliding" project](https://www.v-dem.net/en/our-work/research-projects/pandemic-backsliding/)

    -   The data was downloaded from [V-Dem's GitHub page](https://github.com/vdeminstitute/pandem)

-   Freedom_house.xlsx

    -   This data comes from Freedom House's [Freedom in the World](https://freedomhouse.org/report/freedom-world)
    -   We downloaded "Aggregate Category and Subcategory Scores, 2003-2021" and converted status into 0 (Not Free), 0.5 (Partly Free), and 1 (Free)

-   WVS_TimeSeries_R\_v1_6.rds

    -   This data comes from the World Values Survey

    -   The data was downloaded from [WVS Database (worldvaluessurvey.org)](https://www.worldvaluessurvey.org/WVSDocumentationWVL.jsp)

-   The R package "WDI" is used to access updated data from the World Bank, specifically

    -   Public Health Expenditure Per Capita

    -   GDP

    -   GDP Per Capita

    -   Gini Coefficient

### Additional Data that is imported but not included in current analysis

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

-   vdem_clean.csv

    -   This comes from the [V-Dem Dataset - Version 10](https://www.v-dem.net/en/data/data/v-dem-dataset/).
    -   We downloaded "Country-Year: V-Dem Core" data and merged three indicators (v2x_libdem: Liberal democracy index), v2x_partipdem: Participatory democracy index, v2x_cspart: Civil society participation index)
    -   This data goes back many years. We include data starting in 2006
    -   Version 11 was uploaded [here](https://www.v-dem.net/en/data/data/v-dem-dataset-v11/). Next team can update our analysis using this version.
