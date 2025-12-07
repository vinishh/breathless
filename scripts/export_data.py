import os
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, trim

def export_master_dataset():
    print("Initializing Spark for Data Export...")
    spark = (SparkSession.builder
             .appName("Breathless_Data_Export")
             .master("local[16]")
             .config("spark.driver.memory", "16g")
             .config("spark.sql.shuffle.partitions", "128")
             .getOrCreate())

    print("Loading Data...")
    
    def load_un_data(path, value_col_name, series_filter=None):
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            return None
        try:
            pdf = pd.read_csv(path, header=1)
        except Exception as e:
            return None
        
        pdf = pdf.drop(columns=["Footnotes", "Source"], errors='ignore')
        if 'Value' in pdf.columns:
            pdf['Value'] = pdf['Value'].astype(str)
        else:
            return None
        
        sdf = spark.createDataFrame(pdf)
        
        target_country_col = None
        if "Unnamed: 1" in sdf.columns:
            target_country_col = "Unnamed: 1"
        elif "Region/Country/Area" in sdf.columns:
            sample_val = str(pdf.iloc[0, 0]) if not pdf.empty else ""
            if sample_val.isdigit():
                 if len(sdf.columns) > 1:
                     target_country_col = sdf.columns[1]
            else:
                 target_country_col = "Region/Country/Area"
        else:
            target_country_col = sdf.columns[1] if len(sdf.columns) > 1 else sdf.columns[0]

        clean_sdf = sdf.select(
            trim(col(target_country_col)).alias("Country"),
            col("Year").cast("int"),
            col("Series"),
            regexp_replace(col("Value"), ",", "").cast("double").alias("Value")
        )
        
        if series_filter:
            clean_sdf = clean_sdf.filter(col("Series") == series_filter)
            clean_sdf = clean_sdf.withColumnRenamed("Value", value_col_name).drop("Series")
            
        return clean_sdf

    # Load WHO data
    who_mortality_path = "who attribate deaths per 1000 standarised/data.csv"
    who_df = spark.read.option("header", "true").option("inferSchema", "true").csv(who_mortality_path)
    who_clean_df = who_df.select(
        trim(col("Location")).alias("Country"),
        col("Period").cast("int").alias("Year"),
        col("FactValueNumeric").alias("DeathRate")
    ).filter(col("Dim1") == "Both sexes")

    who_pm25_path = "who pm2.5/dataall.csv"
    who_pm25_df = spark.read.option("header", "true").option("inferSchema", "true").csv(who_pm25_path)
    who_pm25_clean = who_pm25_df.select(
        trim(col("Location")).alias("Country"),
        col("Period").cast("int").alias("Year"),
        col("FactValueNumeric").alias("PM25")
    ).filter(col("Dim1") == "Total")

    # Load UN data
    un_gdp = load_un_data("un data/gdp and gdp per cap/SYB67_230_202411_GDP and GDP Per Capita.csv", "GDP", "GDP in current prices (millions of US dollars)")
    un_gdp_capita = load_un_data("un data/gdp and gdp per cap/SYB67_230_202411_GDP and GDP Per Capita.csv", "GDP_per_capita", "GDP per capita (US dollars)")
    
    un_pop_raw = load_un_data("un data/pop surface area density/SYB67_1_202411_Population, Surface Area and Density.csv", "Value")
    un_pop = un_pop_raw.filter(col("Series") == "Population mid-year estimates (millions)").withColumnRenamed("Value", "Population").drop("Series")
    un_pop_density = un_pop_raw.filter(col("Series") == "Population density").withColumnRenamed("Value", "PopDensity").drop("Series")

    un_demo_raw = load_un_data("un data/pop grrowth, fertility, life expectancy/SYB67_246_202411_Population Growth, Fertility and Mortality Indicators.csv", "Value")
    un_life_exp = un_demo_raw.filter(col("Series") == "Life expectancy at birth for both sexes (years)").withColumnRenamed("Value", "LifeExpectancy").drop("Series")
    un_fertility = un_demo_raw.filter(col("Series") == "Total fertility rate (children per woman)").withColumnRenamed("Value", "Fertility").drop("Series")
    
    un_edu_raw = load_un_data("un data/education at primary, secondary, tertiary/SYB67_309_202411_Education.csv", "Value")
    un_edu_primary = un_edu_raw.filter(col("Series") == "Gross enrollment ratio - Primary (male)").withColumnRenamed("Value", "Education_Primary_Male").drop("Series") 
    
    un_energy = load_un_data("un data/energy/SYB67_263_202411_Production, Trade and Supply of Energy.csv", "Energy_Supply", "Primary energy production (petajoules)") 

    # join the data
    print("Joining Data...")
    master_df = who_clean_df.join(who_pm25_clean, ["Country", "Year"], "inner")
    
    datasets = [un_gdp, un_gdp_capita, un_pop, un_pop_density, un_life_exp, un_fertility, un_edu_primary, un_energy]
    for df in datasets:
        if df is not None:
            master_df = master_df.join(df, ["Country", "Year"], "left")

    # export the data
    print("Exporting Master Dataset to CSV...")
    pdf_master = master_df.toPandas()
    
    # Add Region column in Pandas since it was a python function in notebook
    regions = {
        "Europe": ["germany", "france", "italy", "spain", "uk", "poland", "sweden", "norway", "finland", "denmark", "ireland", "netherlands", "belgium", "austria", "switzerland", "portugal", "greece"],
        "Asia": ["china", "india", "japan", "indonesia", "vietnam", "thailand", "bangladesh", "pakistan", "philippines", "korea", "malaysia", "myanmar"],
        "Africa": ["nigeria", "egypt", "south africa", "kenya", "ethiopia", "tanzania", "ghana", "morocco", "algeria", "uganda", "sudan"],
        "Americas": ["usa", "canada", "brazil", "mexico", "argentina", "colombia", "chile", "peru", "venezuela"],
        "Oceania": ["australia", "new zealand"]
    }
    def get_region(country):
        c = str(country).lower()
        for reg, countries in regions.items():
            for k in countries:
                if k in c:
                    return reg
        return "Other"
        
    pdf_master["Region"] = pdf_master["Country"].apply(get_region)
    
    output_path = "data/processed/master_dataset.csv"
    pdf_master.to_csv(output_path, index=False)
    print(f"Successfully exported {len(pdf_master)} rows to {output_path}")

if __name__ == "__main__":
    export_master_dataset()
