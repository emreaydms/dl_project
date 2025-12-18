"""
Weather Data Fetcher - Hungary Temperature Data (15-minute resolution)

Macaristan'ın en yoğun nüfuslu 5 şehrindeki sıcaklık verisini çeker,
nüfusa göre ağırlıklı ortalama hesaplar ve HDD/CDD değerlerini üretir.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import logging
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WeatherDataFetcher:
    """Macaristan şehirleri için hava durumu verisi çekici"""
    
    # Macaristan'ın en yoğun nüfuslu 5 şehri ve koordinatları (lat, lon)
    CITIES = {
        'Budapest': {
            'lat': 47.4979,
            'lon': 19.0402,
            'population': 1750000
        },
        'Debrecen': {
            'lat': 47.5316,
            'lon': 21.6273,
            'population': 200000
        },
        'Szeged': {
            'lat': 46.2530,
            'lon': 20.1414,
            'population': 160000
        },
        'Miskolc': {
            'lat': 48.1034,
            'lon': 20.7784,
            'population': 150000
        },
        'Pecs': {
            'lat': 46.0727,
            'lon': 18.2328,
            'population': 140000
        }
    }
    
    # HDD/CDD eşik değerleri
    HDD_BASE_TEMP = 18.0  # °C - Referans iç ortam sıcaklığı
    HDD_THRESHOLD = 15.0  # °C - Isıtma eşiği
    CDD_THRESHOLD = 22.0  # °C - Soğutma eşiği
    
    def __init__(
        self,
        base_url: str = "https://archive-api.open-meteo.com/v1/archive",
        timezone: str = "Europe/Budapest"
    ):
        self.base_url = base_url
        self.local_tz = pytz.timezone(timezone)
        self.utc_tz = pytz.UTC
        
        # Toplam nüfusu hesapla (ağırlıklı ortalama için)
        self.total_population = sum(city['population'] for city in self.CITIES.values())
        
        # Retry mekanizması
        self.session = requests.Session()
        retry = Retry(connect=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def fetch_city_temperature(self, city_name: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Belirli bir şehir için sıcaklık verisini çek (saatlik)
        
        Args:
            city_name: Şehir adı (CITIES dict'inde olmalı)
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
        
        Returns:
            DataFrame with columns: datetime, temperature_2m (UTC timezone)
        """
        if city_name not in self.CITIES:
            raise ValueError(f"Şehir bulunamadı: {city_name}. Mevcut şehirler: {list(self.CITIES.keys())}")
        
        city = self.CITIES[city_name]
        
        # Tarihleri UTC'ye çevir
        if start_date.tzinfo is None:
            start_date = self.local_tz.localize(start_date)
        if end_date.tzinfo is None:
            end_date = self.local_tz.localize(end_date)
        
        start_date_utc = start_date.astimezone(self.utc_tz)
        end_date_utc = end_date.astimezone(self.utc_tz)
        
        # API formatı: YYYY-MM-DD
        # Son günün 23:00'ına kadar veri çekmek için end_date'i bir sonraki gün olarak gönder
        # (API inclusive değil, bir sonraki günün başlangıcını kullan)
        start_str = start_date_utc.strftime('%Y-%m-%d')
        end_date_for_api = end_date_utc + timedelta(days=1)
        end_str = end_date_for_api.strftime('%Y-%m-%d')
        
        params = {
            "latitude": city['lat'],
            "longitude": city['lon'],
            "start_date": start_str,
            "end_date": end_str,
            "hourly": "temperature_2m",
            "timezone": "UTC"
        }
        
        all_data = []
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'hourly' in data:
                    times = data['hourly']['time']
                    temps = data['hourly']['temperature_2m']
                    
                    for t, temp in zip(times, temps):
                        if temp is not None:
                            # API'den gelen datetime'ı UTC olarak parse et
                            dt = pd.to_datetime(t)
                            # Eğer timezone-aware değilse UTC olarak işaretle
                            if dt.tz is None:
                                dt = dt.tz_localize('UTC')
                            else:
                                dt = dt.tz_convert('UTC')
                            
                            # Sadece istenen tarih aralığındaki verileri al
                            # Başlangıç: start_date_utc (00:00)
                            # Bitiş: end_date_utc'in 23:00'ı (23:00 dahil)
                            start_filter = start_date_utc.replace(hour=0, minute=0, second=0, microsecond=0)
                            end_filter = end_date_utc.replace(hour=23, minute=0, second=0, microsecond=0)
                            
                            if start_filter <= dt <= end_filter:
                                all_data.append({
                                    'datetime': dt,
                                    'temperature_2m': float(temp)
                                })
                    
                    logger.info(f"✓ {city_name}: {len(all_data)} saatlik kayıt")
                else:
                    logger.warning(f"⚠ {city_name}: Veri yok")
            else:
                logger.error(f"❌ {city_name}: HTTP {response.status_code}")
        
        except Exception as e:
            logger.error(f"❌ {city_name}: {e}")
        
        if not all_data:
            return pd.DataFrame(columns=['datetime', 'temperature_2m'])
        
        df = pd.DataFrame(all_data)
        df = df.sort_values('datetime').drop_duplicates(subset=['datetime'], keep='last')
        
        return df
    
    def fetch_all_cities(self, start_date: datetime, end_date: datetime, show_progress: bool = True) -> dict:
        """
        Tüm şehirler için sıcaklık verisini çek
        
        Returns:
            Dict: {city_name: DataFrame} - Her DataFrame UTC timezone'da
        """
        all_city_data = {}
        
        if show_progress:
            print(f"\n🌡️ {len(self.CITIES)} şehir için sıcaklık verisi çekiliyor...\n")
        
        for idx, city_name in enumerate(self.CITIES.keys(), 1):
            if show_progress:
                print(f"[{idx}/{len(self.CITIES)}] {city_name}...", end=" ", flush=True)
            
            city_data = self.fetch_city_temperature(city_name, start_date, end_date)
            
            if not city_data.empty:
                all_city_data[city_name] = city_data
                if show_progress:
                    print(f"✓ {len(city_data)} kayıt")
            else:
                if show_progress:
                    print("⚠ Veri yok")
            
            time.sleep(0.5)  # Rate limiting
        
        if show_progress:
            print()
        
        return all_city_data
    
    def calculate_weighted_average_temperature(self, city_data_dict: dict) -> pd.DataFrame:
        """
        Nüfusa göre ağırlıklı ortalama sıcaklık hesapla
        
        Args:
            city_data_dict: {city_name: DataFrame} formatında şehir verileri (UTC timezone)
        
        Returns:
            DataFrame with columns: datetime, temperature_2m (weighted average, UTC timezone)
        """
        if not city_data_dict:
            return pd.DataFrame(columns=['datetime', 'temperature_2m'])
        
        # Tüm şehirlerin datetime'larını birleştir ve UTC'ye normalize et
        normalized_city_data = {}
        all_datetimes = set()
        
        for city_name, df in city_data_dict.items():
            if df.empty or 'datetime' not in df.columns:
                continue
            
            # Datetime'ları UTC'ye normalize et
            df_normalized = df.copy()
            df_normalized['datetime'] = pd.to_datetime(df_normalized['datetime'])
            
            # Timezone kontrolü ve normalize
            if df_normalized['datetime'].dt.tz is None:
                df_normalized['datetime'] = df_normalized['datetime'].dt.tz_localize('UTC')
            else:
                df_normalized['datetime'] = df_normalized['datetime'].dt.tz_convert('UTC')
            
            normalized_city_data[city_name] = df_normalized
            all_datetimes.update(df_normalized['datetime'].values)
        
        if not all_datetimes:
            logger.warning("Hiç datetime bulunamadı")
            return pd.DataFrame(columns=['datetime', 'temperature_2m'])
        
        # Tüm datetime'ları birleştir ve merge ile birleştir
        # Önce tüm şehir verilerini birleştir
        merged_df = None
        
        for city_name, df_normalized in normalized_city_data.items():
            df_copy = df_normalized[['datetime', 'temperature_2m']].copy()
            # Sütun adını şehir adıyla değiştir
            df_copy = df_copy.rename(columns={'temperature_2m': f'temp_{city_name}'})
            df_copy = df_copy.set_index('datetime')
            
            if merged_df is None:
                merged_df = df_copy
            else:
                merged_df = merged_df.join(df_copy, how='outer')
        
        if merged_df is None or merged_df.empty:
            logger.warning("Birleştirilmiş DataFrame oluşturulamadı")
            return pd.DataFrame(columns=['datetime', 'temperature_2m'])
        
        # Her satır için ağırlıklı ortalama hesapla
        result_data = []
        
        for dt_utc, row in merged_df.iterrows():
            weighted_sum = 0.0
            total_weight = 0.0
            
            for city_name in normalized_city_data.keys():
                temp_col = f'temp_{city_name}'
                if temp_col in row.index:
                    temp = row[temp_col]
                    if pd.notna(temp) and not np.isnan(temp):
                        weight = self.CITIES[city_name]['population']
                        weighted_sum += float(temp) * weight
                        total_weight += weight
            
            if total_weight > 0:
                avg_temp = weighted_sum / total_weight
                result_data.append({
                    'datetime': dt_utc,
                    'temperature_2m': avg_temp
                })
        
        if not result_data:
            logger.warning(f"Ağırlıklı ortalama hesaplanamadı. {len(merged_df)} datetime için {len(normalized_city_data)} şehir verisi var.")
            return pd.DataFrame(columns=['datetime', 'temperature_2m'])
        
        result_df = pd.DataFrame(result_data)
        
        if result_df.empty or 'datetime' not in result_df.columns:
            logger.warning("Sonuç DataFrame boş veya datetime sütunu yok")
            return pd.DataFrame(columns=['datetime', 'temperature_2m'])
        
        result_df = result_df.sort_values('datetime').reset_index(drop=True)
        
        logger.info(f"Ağırlıklı ortalama hesaplandı: {len(result_df)} kayıt")
        
        return result_df
    
    def resample_to_15min(self, df: pd.DataFrame, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Saatlik veriyi 15 dakikalığa indir (interpolasyon ile)
        
        Args:
            df: Saatlik sıcaklık DataFrame'i (UTC timezone)
            start_date: Başlangıç tarihi (opsiyonel, verilmezse çekilen verinin min'i kullanılır)
            end_date: Bitiş tarihi (opsiyonel, verilmezse çekilen verinin max'ı kullanılır)
        
        Returns:
            15 dakikalık çözünürlükte DataFrame (UTC timezone)
        """
        if df.empty:
            return pd.DataFrame(columns=['datetime', 'temperature_2m'])
        
        df = df.copy()
        df = df.sort_values('datetime').drop_duplicates(subset=['datetime'], keep='last')
        
        # Datetime'ları UTC'ye normalize et
        df['datetime'] = pd.to_datetime(df['datetime'])
        if df['datetime'].dt.tz is None:
            df['datetime'] = df['datetime'].dt.tz_localize('UTC')
        else:
            df['datetime'] = df['datetime'].dt.tz_convert('UTC')
        
        df = df.set_index('datetime')
        
        # Tarih aralığını belirle
        if start_date is None or end_date is None:
            # Çekilen verinin tarih aralığını kullan
            min_date = df.index.min()
            max_date = df.index.max()
            start_date = min_date.replace(minute=0, second=0, microsecond=0)
            end_date = max_date.replace(minute=45, second=0, microsecond=0)
        else:
            # Kullanıcının belirttiği tarih aralığını kullan
            # Eğer timezone-aware değilse, direkt UTC olarak kabul et
            if start_date.tzinfo is None:
                start_date = pd.Timestamp(start_date).tz_localize('UTC')
            else:
                start_date = pd.Timestamp(start_date).tz_convert('UTC')
            
            if end_date.tzinfo is None:
                end_date = pd.Timestamp(end_date).tz_localize('UTC')
            else:
                end_date = pd.Timestamp(end_date).tz_convert('UTC')
            
            # Başlangıç: İlk günün başlangıcı (00:00)
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            # Bitiş: Son günün sonu (23:45)
            end_date = end_date.replace(hour=23, minute=45, second=0, microsecond=0)
        
        # 15 dakikalık zaman ızgarası oluştur (UTC)
        full_idx = pd.date_range(
            start=start_date,
            end=end_date,
            freq='15min',
            tz='UTC'
        )
        
        # Veriyi bu ızgaraya oturt ve interpolasyon yap
        df_resampled = df.reindex(full_idx)
        
        # temperature_2m sütunu var mı kontrol et
        if 'temperature_2m' not in df_resampled.columns:
            logger.warning(f"resample_to_15min: 'temperature_2m' sütunu yok. Mevcut sütunlar: {df_resampled.columns.tolist()}")
            return pd.DataFrame(columns=['datetime', 'temperature_2m'])
        
        df_resampled['temperature_2m'] = df_resampled['temperature_2m'].interpolate(
            method='linear', limit_direction='both'
        )
        
        df_resampled.index.name = 'datetime'
        df_resampled = df_resampled.reset_index()
        
        return df_resampled
    
    def calculate_daily_hdd_cdd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Günlük HDD ve CDD değerlerini hesapla
        
        HDD (Heating Degree Days):
        - Günlük ortalama sıcaklık <= 15°C ise: HDD = 18°C - T_avg
        - Günlük ortalama sıcaklık > 15°C ise: HDD = 0
        
        CDD (Cooling Degree Days):
        - Günlük ortalama sıcaklık > 22°C ise: CDD = T_avg - 22°C
        - Günlük ortalama sıcaklık <= 22°C ise: CDD = 0
        
        Args:
            df: 15 dakikalık sıcaklık DataFrame'i (datetime, temperature_2m) - UTC timezone
        
        Returns:
            DataFrame with columns: datetime (daily, UTC), temperature_2m (daily avg), hdd, cdd
        """
        if df.empty:
            return pd.DataFrame(columns=['datetime', 'temperature_2m', 'hdd', 'cdd'])
        
        if 'temperature_2m' not in df.columns:
            logger.error(f"calculate_daily_hdd_cdd: 'temperature_2m' sütunu yok. Mevcut sütunlar: {list(df.columns)}")
            return pd.DataFrame(columns=['datetime', 'temperature_2m', 'hdd', 'cdd'])
        
        # DataFrame'i kopyala ve temizle
        df_clean = df.copy()
        
        # datetime sütununu kontrol et ve normalize et
        if 'datetime' not in df_clean.columns:
            logger.error("calculate_daily_hdd_cdd: 'datetime' sütunu yok")
            return pd.DataFrame(columns=['datetime', 'temperature_2m', 'hdd', 'cdd'])
        
        df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])
        
        # Timezone kontrolü ve normalize
        if df_clean['datetime'].dt.tz is None:
            df_clean['datetime'] = df_clean['datetime'].dt.tz_localize('UTC')
        else:
            df_clean['datetime'] = df_clean['datetime'].dt.tz_convert('UTC')
        
        # NaN değerleri temizle
        df_clean = df_clean.dropna(subset=['temperature_2m'])
        
        if df_clean.empty:
            logger.warning("calculate_daily_hdd_cdd: Tüm sıcaklık değerleri NaN")
            return pd.DataFrame(columns=['datetime', 'temperature_2m', 'hdd', 'cdd'])
        
        # Index olarak datetime kullan
        df_clean = df_clean.set_index('datetime')
        
        # Günlük ortalama sıcaklık hesapla
        try:
            daily_avg = df_clean['temperature_2m'].resample('D').mean()
        except Exception as e:
            logger.error(f"calculate_daily_hdd_cdd: Resample hatası: {e}")
            return pd.DataFrame(columns=['datetime', 'temperature_2m', 'hdd', 'cdd'])
        
        if len(daily_avg) == 0:
            logger.warning("calculate_daily_hdd_cdd: Günlük ortalama hesaplanamadı")
            return pd.DataFrame(columns=['datetime', 'temperature_2m', 'hdd', 'cdd'])
        
        # HDD ve CDD hesapla
        hdd = np.where(
            daily_avg <= self.HDD_THRESHOLD,
            self.HDD_BASE_TEMP - daily_avg,
            0.0
        )
        
        cdd = np.where(
            daily_avg > self.CDD_THRESHOLD,
            daily_avg - self.CDD_THRESHOLD,
            0.0
        )
        
        # DataFrame oluştur
        result_df = pd.DataFrame({
            'datetime': daily_avg.index,
            'temperature_2m': daily_avg.values,
            'hdd': hdd,
            'cdd': cdd
        })
        
        # datetime sütununu reset et (index'ten sütuna)
        result_df = result_df.reset_index(drop=True)
        
        logger.info(f"calculate_daily_hdd_cdd: {len(result_df)} günlük kayıt hesaplandı")
        
        return result_df
    
    def calculate_weighted_hdd_cdd(self, city_data_dict: dict) -> pd.DataFrame:
        """
        Her şehir için günlük HDD/CDD hesapla, sonra nüfusa göre ağırlıklı ortalama al
        
        Args:
            city_data_dict: {city_name: DataFrame} formatında şehir verileri (15 dakikalık, UTC)
        
        Returns:
            DataFrame with columns: datetime (daily, UTC), hdd (weighted), cdd (weighted)
        """
        if not city_data_dict:
            logger.warning("calculate_weighted_hdd_cdd: Boş city_data_dict")
            return pd.DataFrame(columns=['datetime', 'hdd', 'cdd'])
        
        # Her şehir için günlük HDD/CDD hesapla
        city_daily_hdd_cdd = {}
        
        for city_name, df in city_data_dict.items():
            if df.empty:
                logger.warning(f"{city_name}: Boş DataFrame, HDD/CDD hesaplanamıyor")
                continue
            
            if 'temperature_2m' not in df.columns:
                logger.warning(f"{city_name}: 'temperature_2m' sütunu yok")
                continue
            
            try:
                daily_df = self.calculate_daily_hdd_cdd(df)
                if not daily_df.empty and 'hdd' in daily_df.columns and 'cdd' in daily_df.columns:
                    city_daily_hdd_cdd[city_name] = daily_df
                    logger.info(f"{city_name}: {len(daily_df)} günlük HDD/CDD hesaplandı")
                else:
                    logger.warning(f"{city_name}: Günlük HDD/CDD DataFrame boş veya eksik sütunlar")
            except Exception as e:
                logger.error(f"{city_name}: HDD/CDD hesaplama hatası: {e}")
                continue
        
        if not city_daily_hdd_cdd:
            logger.error(f"Hiç şehir için HDD/CDD hesaplanamadı. {len(city_data_dict)} şehir verisi var.")
            return pd.DataFrame(columns=['datetime', 'hdd', 'cdd'])
        
        # Tüm günleri birleştir - merge kullanarak daha güvenilir
        merged_df = None
        
        for city_name, daily_df in city_daily_hdd_cdd.items():
            df_copy = daily_df[['datetime', 'hdd', 'cdd']].copy()
            df_copy = df_copy.rename(columns={'hdd': f'hdd_{city_name}', 'cdd': f'cdd_{city_name}'})
            df_copy = df_copy.set_index('datetime')
            
            if merged_df is None:
                merged_df = df_copy
            else:
                merged_df = merged_df.join(df_copy, how='outer')
        
        if merged_df is None or merged_df.empty:
            logger.error("Birleştirilmiş HDD/CDD DataFrame oluşturulamadı")
            return pd.DataFrame(columns=['datetime', 'hdd', 'cdd'])
        
        # Her gün için ağırlıklı ortalama HDD/CDD hesapla
        weighted_results = []
        
        for dt, row in merged_df.iterrows():
            hdd_weighted_sum = 0.0
            cdd_weighted_sum = 0.0
            total_weight = 0.0
            
            for city_name in city_daily_hdd_cdd.keys():
                hdd_col = f'hdd_{city_name}'
                cdd_col = f'cdd_{city_name}'
                
                if hdd_col in row.index and cdd_col in row.index:
                    hdd = row[hdd_col]
                    cdd = row[cdd_col]
                    
                    if pd.notna(hdd) and pd.notna(cdd) and not np.isnan(hdd) and not np.isnan(cdd):
                        weight = self.CITIES[city_name]['population']
                        hdd_weighted_sum += float(hdd) * weight
                        cdd_weighted_sum += float(cdd) * weight
                        total_weight += weight
            
            if total_weight > 0:
                weighted_results.append({
                    'datetime': dt,
                    'hdd': hdd_weighted_sum / total_weight,
                    'cdd': cdd_weighted_sum / total_weight
                })
        
        if not weighted_results:
            logger.error("Ağırlıklı HDD/CDD hesaplanamadı")
            return pd.DataFrame(columns=['datetime', 'hdd', 'cdd'])
        
        result_df = pd.DataFrame(weighted_results)
        result_df = result_df.sort_values('datetime').reset_index(drop=True)
        
        logger.info(f"calculate_weighted_hdd_cdd: {len(result_df)} günlük ağırlıklı HDD/CDD hesaplandı")
        
        return result_df
    
    def expand_daily_hdd_cdd_to_15min(self, daily_hdd_cdd_df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Günlük HDD/CDD değerlerini 15 dakikalık çözünürlüğe genişlet
        (Aynı gün içinde sabit değerler)
        
        Args:
            daily_hdd_cdd_df: Günlük HDD/CDD DataFrame'i (UTC timezone)
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
        
        Returns:
            15 dakikalık çözünürlükte DataFrame (datetime, hdd, cdd) - UTC timezone
        """
        if daily_hdd_cdd_df.empty:
            return pd.DataFrame(columns=['datetime', 'hdd', 'cdd'])
        
        # Tarihleri UTC'ye çevir
        # Eğer timezone-aware değilse, direkt UTC olarak kabul et
        if start_date.tzinfo is None:
            start_date_utc = pd.Timestamp(start_date).tz_localize('UTC')
        else:
            start_date_utc = pd.Timestamp(start_date).tz_convert('UTC')
        
        if end_date.tzinfo is None:
            end_date_utc = pd.Timestamp(end_date).tz_localize('UTC')
        else:
            end_date_utc = pd.Timestamp(end_date).tz_convert('UTC')
        
        # Başlangıç: İlk günün başlangıcı (00:00)
        start_date_utc = start_date_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        # Bitiş: Son günün sonu (23:45)
        end_date_utc = end_date_utc.replace(hour=23, minute=45, second=0, microsecond=0)
        
        # 15 dakikalık zaman ızgarası oluştur
        full_idx = pd.date_range(
            start=start_date_utc,
            end=end_date_utc,
            freq='15min',
            tz='UTC'
        )
        
        # Günlük veriyi datetime index'e çevir ve UTC'ye normalize et
        daily_hdd_cdd_df = daily_hdd_cdd_df.copy()
        daily_hdd_cdd_df['datetime'] = pd.to_datetime(daily_hdd_cdd_df['datetime'])
        
        # Datetime'ları UTC'ye normalize et
        if daily_hdd_cdd_df['datetime'].dt.tz is None:
            daily_hdd_cdd_df['datetime'] = daily_hdd_cdd_df['datetime'].dt.tz_localize('UTC')
        else:
            daily_hdd_cdd_df['datetime'] = daily_hdd_cdd_df['datetime'].dt.tz_convert('UTC')
        
        daily_hdd_cdd_df = daily_hdd_cdd_df.set_index('datetime')
        
        # Her gün için HDD/CDD değerlerini 15 dakikalık aralıklara genişlet
        result_data = []
        
        # Günlük veriyi date ile eşleştirmek için dict oluştur
        daily_dict = {}
        for idx, row in daily_hdd_cdd_df.iterrows():
            # Index'i date'e çevir (timezone bilgisini kaldır, sadece tarih)
            if isinstance(idx, pd.Timestamp):
                date_key = idx.date()
            else:
                date_key = pd.to_datetime(idx).date()
            daily_dict[date_key] = {'hdd': row['hdd'], 'cdd': row['cdd']}
        
        for dt in full_idx:
            # Bu datetime'ın ait olduğu günü bul
            day_date = dt.date()
            
            # Günlük veriden bu günün HDD/CDD değerlerini al
            if day_date in daily_dict:
                hdd = daily_dict[day_date]['hdd']
                cdd = daily_dict[day_date]['cdd']
            else:
                hdd = 0.0
                cdd = 0.0
            
            result_data.append({
                'datetime': dt,
                'hdd': hdd,
                'cdd': cdd
            })
        
        result_df = pd.DataFrame(result_data)
        result_df = result_df.sort_values('datetime').reset_index(drop=True)
        
        return result_df
    
    def fetch_and_process_temperature(self, start_date: datetime, end_date: datetime, show_progress: bool = True) -> pd.DataFrame:
        """
        Sadece sıcaklık verisini çek ve işle: Veri çek, ağırlıklı ortalama hesapla, 15 dakikalığa indir
        
        Args:
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            show_progress: İlerleme mesajlarını göster
        
        Returns:
            (temperature_df, city_data) - 15 dakikalık çözünürlükte ağırlıklı ortalama sıcaklık ve şehir verileri (HDD/CDD için kullanılabilir)
        """
        # 1. Tüm şehirler için sıcaklık verisini çek
        city_data = self.fetch_all_cities(start_date, end_date, show_progress)
        
        if not city_data:
            logger.warning("Hiç şehir verisi çekilemedi")
            return pd.DataFrame(columns=['datetime', 'temperature_2m']), {}
        
        # 2. Nüfusa göre ağırlıklı ortalama sıcaklık hesapla (saatlik)
        if show_progress:
            print("\n📊 Nüfusa göre ağırlıklı ortalama sıcaklık hesaplanıyor...")
        
        weighted_temp_hourly = self.calculate_weighted_average_temperature(city_data)
        
        if weighted_temp_hourly.empty:
            logger.warning("Ağırlıklı ortalama sıcaklık hesaplanamadı")
            return pd.DataFrame(columns=['datetime', 'temperature_2m']), city_data
        
        # 3. Saatlik veriyi 15 dakikalığa indir (kullanıcının belirttiği tarih aralığını kullan)
        if show_progress:
            print("🔄 Saatlik veri 15 dakikalığa indiriliyor...")
        
        weighted_temp_15min = self.resample_to_15min(weighted_temp_hourly, start_date, end_date)
        
        if show_progress:
            print(f"\n✅ Sıcaklık verisi işlendi!")
            print(f"   Toplam kayıt sayısı: {len(weighted_temp_15min):,}\n")
        
        return weighted_temp_15min, city_data
    
    def fetch_and_process_hdd_cdd(self, start_date: datetime, end_date: datetime, city_data: dict = None, show_progress: bool = True) -> pd.DataFrame:
        """
        Sadece HDD/CDD verisini çek ve işle: Veri çek, günlük HDD/CDD hesapla, 15 dakikalığa genişlet
        
        Args:
            start_date: Başlangıç tarihi
            end_date: Bitiş tarihi
            city_data: Opsiyonel - Eğer verilirse, bu veriyi kullanır (tekrar çekmez)
            show_progress: İlerleme mesajlarını göster
        
        Returns:
            hdd_cdd_df - 15 dakikalık çözünürlükte HDD/CDD (günlük sabit değerler, UTC timezone)
        """
        # 1. Eğer city_data verilmemişse, tüm şehirler için sıcaklık verisini çek
        if city_data is None:
            city_data = self.fetch_all_cities(start_date, end_date, show_progress)
        
        if not city_data:
            logger.warning("Hiç şehir verisi çekilemedi")
            return pd.DataFrame(columns=['datetime', 'hdd', 'cdd'])
        
        # 2. Her şehir için saatlik veriyi 15 dakikalığa indir
        if show_progress:
            print("\n🔄 Şehir verileri 15 dakikalığa indiriliyor...")
        
        city_data_15min = {}
        for city_name, hourly_df in city_data.items():
            resampled_df = self.resample_to_15min(hourly_df, start_date, end_date)
            if not resampled_df.empty:
                city_data_15min[city_name] = resampled_df
            else:
                logger.warning(f"{city_name}: 15 dakikalık veri boş")
        
        if not city_data_15min:
            logger.warning("Hiç şehir için 15 dakikalık veri oluşturulamadı")
            return pd.DataFrame(columns=['datetime', 'hdd', 'cdd'])
        
        # 3. Her şehir için günlük HDD/CDD hesapla ve ağırlıklı ortalama al
        if show_progress:
            print("🌡️ Günlük HDD/CDD değerleri hesaplanıyor...")
        
        daily_hdd_cdd = self.calculate_weighted_hdd_cdd(city_data_15min)
        
        if daily_hdd_cdd.empty:
            logger.warning("HDD/CDD değerleri hesaplanamadı")
            return pd.DataFrame(columns=['datetime', 'hdd', 'cdd'])
        
        # 4. Günlük HDD/CDD'yi 15 dakikalığa genişlet
        if show_progress:
            print("📅 Günlük HDD/CDD değerleri 15 dakikalığa genişletiliyor...")
        
        hdd_cdd_15min = self.expand_daily_hdd_cdd_to_15min(daily_hdd_cdd, start_date, end_date)
        
        if show_progress:
            print(f"\n✅ HDD/CDD verisi işlendi!")
            print(f"   Toplam kayıt sayısı: {len(hdd_cdd_15min):,}\n")
        
        return hdd_cdd_15min
    
    def save_csv(self, df: pd.DataFrame, filepath: str):
        """CSV'ye kaydet"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"✅ Veri kaydedildi: {filepath}")


if __name__ == "__main__":
    fetcher = WeatherDataFetcher()
    
    start = datetime(2015, 1, 1)
    end = datetime(2024, 12, 31)
    
    # Sıcaklık verisini çek ve işle
    temp_df = fetcher.fetch_and_process_temperature(start, end)
    fetcher.save_csv(temp_df, "data/raw/hungary_temperature_2015_2024.csv")
    
    # HDD/CDD verisini çek ve işle
    hdd_cdd_df = fetcher.fetch_and_process_hdd_cdd(start, end)
    fetcher.save_csv(hdd_cdd_df, "data/raw/hungary_hdd_cdd_2015_2024.csv")


