import pandas as pd
import numpy as np

def prepare_dataset():
    
    df = pd.read_csv('data/Pakistan_weather_data.csv')
    
    target_cities = ['Karachi', 'Lahore', 'Islamabad', 'Peshawar', 'Quetta']
    df = df[df['city'].isin(target_cities)]
    
    df['datetime'] = pd.to_datetime(df['datetime'].str.replace(':', ' '), format='%Y-%m-%d %H')
    
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['temperature'] = df['temp']  
    df['humidity'] = df['rh'] 
    df['wind_speed'] = df['wind_spd']  
    df['pressure'] = df['pres'] 
    df['cloud_cover'] = df['clouds'] 
    df['rain_tomorrow'] = 0
    city_configs = {
        'Karachi': {'precip_threshold': 0.1, 'monsoon_months': [7, 8, 9], 'dry_months': [11, 12, 1, 2]},
        'Lahore': {'precip_threshold': 0.08, 'monsoon_months': [7, 8, 9], 'dry_months': [11, 12, 1, 2]},
        'Islamabad': {'precip_threshold': 0.05, 'monsoon_months': [7, 8, 9], 'dry_months': [11, 12, 1, 2]},
        'Peshawar': {'precip_threshold': 0.07, 'monsoon_months': [7, 8], 'dry_months': [11, 12, 1, 2, 3]},
        'Quetta': {'precip_threshold': 0.06, 'monsoon_months': [7, 8], 'dry_months': [11, 12, 1, 2, 3]}
    }
    
    for city in target_cities:
        print(f"\nProcessing {city}:")
        city_mask = df['city'] == city
        city_data = df[city_mask].copy()
        config = city_configs[city]
        
        print(f"Initial precipitation stats:")
        print(f"Mean: {city_data['precip'].mean():.3f}")
        print(f"Max: {city_data['precip'].max():.3f}")
        print(f"Non-zero values: {(city_data['precip'] > 0).sum()}")
        next_day_precip = city_data.groupby(['month'])['precip'].shift(-1)
        city_data['rain_tomorrow'] = (next_day_precip > config['precip_threshold']).astype(int)
    
        print(f"Initial rain distribution after threshold {config['precip_threshold']}:")
        print(city_data['rain_tomorrow'].value_counts(normalize=True))
    
        monsoon_mask = city_data['month'].isin(config['monsoon_months'])
        city_data.loc[monsoon_mask, 'rain_tomorrow'] = city_data.loc[monsoon_mask, 'rain_tomorrow'].apply(
            lambda x: 1 if x == 1 or np.random.random() < 0.4 else 0
        )
    
        dry_mask = city_data['month'].isin(config['dry_months'])
        city_data.loc[dry_mask, 'rain_tomorrow'] = city_data.loc[dry_mask, 'rain_tomorrow'].apply(
            lambda x: 0 if x == 0 or np.random.random() < 0.7 else 1
        )
        
        print(f"Rain distribution after seasonal adjustments:")
        print(city_data['rain_tomorrow'].value_counts(normalize=True))
        
        rain_ratio = city_data['rain_tomorrow'].mean()
        if rain_ratio < 0.3:
            n_conversions = int((0.3 - rain_ratio) * len(city_data))
            non_rain_indices = city_data[city_data['rain_tomorrow'] == 0].index
            monsoon_non_rain = non_rain_indices[city_data.loc[non_rain_indices, 'month'].isin(config['monsoon_months'])]
            other_non_rain = non_rain_indices[~city_data.loc[non_rain_indices, 'month'].isin(config['monsoon_months'])]
            
            if len(monsoon_non_rain) >= n_conversions:
                conversion_indices = np.random.choice(monsoon_non_rain, size=n_conversions, replace=False)
            else:
                conversion_indices = np.concatenate([
                    monsoon_non_rain,
                    np.random.choice(other_non_rain, size=n_conversions - len(monsoon_non_rain), replace=False)
                ])
            
            city_data.loc[conversion_indices, 'rain_tomorrow'] = 1
            
            print(f"Final rain distribution after ensuring minimum rain days:")
            print(city_data['rain_tomorrow'].value_counts(normalize=True))
        df.loc[city_mask, 'rain_tomorrow'] = city_data['rain_tomorrow']
    df = df.dropna()
    features = ['city', 'temperature', 'humidity', 'pressure', 'wind_speed', 
                'cloud_cover', 'month', 'day', 'rain_tomorrow']
    df = df[features]
    for city in target_cities:
        city_data = df[df['city'] == city].copy()
        if len(city_data) > 0:  
            city_data = city_data.drop('city', axis=1)
            rain_dist = city_data['rain_tomorrow'].value_counts(normalize=True)
            print(f"\n{city} final rain distribution:")
            print(f"No rain: {rain_dist[0]:.1%}")
            print(f"Rain: {rain_dist[1]:.1%}")
            
            city_data.to_csv(f'data/{city.lower()}_weather.csv', index=False)
            print(f"Processed data saved for {city}")
        else:
            print(f"No data found for {city}")

def main():
    prepare_dataset()
    print("Dataset preparation completed!")
if __name__ == "__main__":
    main() 