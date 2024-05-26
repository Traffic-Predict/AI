import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from termcolor import colored

# debug = True
debug = False

def print_step(message):
    print(colored(message, 'yellow'))

# 단계별 실행 시간을 측정하기 위한 데코레이터
def timed_step(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print_step(f"{func.__name__} 실행 시간: {end_time - start_time:.2f}초")
        return result
    return wrapper

@timed_step
def load_data_chunk(filename, chunksize):
    # CSV 파일을 청크 단위로 읽기
    tags = ["date", "time", "nodeid", "linkid", "speed", "etc"]
    return pd.read_csv(filename, chunksize=chunksize, names=tags, dtype={'time': str})

@timed_step
def transform_data(data):
    # 'date'와 'time' 열을 개별적으로 datetime으로 변환
    data['date'] = pd.to_datetime(data['date'], format="%Y%m%d")

    # 'time' 열을 변환하는 함수 정의
    def convert_time(time_str):
        try:
            return pd.to_datetime(time_str, format="%H%M").time()
        except ValueError:
            return None

    # 'time' 열 변환
    data['time'] = data['time'].apply(convert_time)

    # 'datetime' 열 생성, 'time' 열에 None이 있는 경우를 처리
    data['datetime'] = data.apply(lambda row: pd.Timestamp.combine(row['date'], row['time']) if row['time'] is not None else pd.NaT, axis=1)

    # 'date'와 'time' 열 삭제
    data = data.drop(["date", "time"], axis=1)

    # 'linkid'와 'nodeid' 열의 NaN 값을 0으로 대체
    data['linkid'] = data['linkid'].fillna(0)
    data['nodeid'] = data['nodeid'].fillna(0)

    # 'linkid'와 'nodeid' 열의 데이터 타입을 int64로 변환
    data['linkid'] = data['linkid'].astype('int64')
    data['nodeid'] = data['nodeid'].astype('int64')

    # 'datetime' 열을 맨 앞으로 이동
    cols = ['datetime'] + [col for col in data if col != 'datetime']
    data = data[cols]

    # 'datetime' 열의 결측치를 제거
    data = data.dropna(subset=['datetime'])
    return data

@timed_step
def save_grouped_data(data, group_by, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    grouped = data.groupby(group_by)

    for group_id, group in grouped:
        group.set_index('datetime', inplace=True)
        resampled_group = group['speed'].resample('5min').mean().reset_index()

        output_file = os.path.join(output_dir, f'{group_by}_{int(group_id)}.csv')

        if os.path.exists(output_file):
            existing_data = pd.read_csv(output_file, parse_dates=['datetime'])
            combined_data = pd.concat([existing_data, resampled_group]).drop_duplicates(subset=['datetime'], keep='last')
        else:
            combined_data = resampled_group

        combined_data.to_csv(output_file, index=False, header=True)
        print_step(f"Saved {output_file}")

@timed_step
def visualize_data(data):
    data.set_index('datetime', inplace=True)
    resampled_data = data['speed'].resample('5min').mean()

    plt.figure(figsize=(10, 6))
    plt.plot(resampled_data.index, resampled_data, label='Average Speed (5 min interval)')
    plt.xlabel('Datetime')
    plt.ylabel('Average Speed')
    plt.title('Average Speed over Time (5 min intervals)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    start_time = time.time()
    print_step("데이터 로드 및 변환 중...")

    filename = "sources/20240101_20240401.csv"
    chunksize = 10000000

    chunk_reader = load_data_chunk(filename, chunksize)

    for chunk in chunk_reader:
        data = transform_data(chunk)

        if debug:
            print_step("변환된 데이터:")
            print(data.head())
            print(data.info())

        print_step("linkid 별로 데이터 저장 중...")
        save_grouped_data(data, 'linkid', "sources/linkid_splits")

        print_step("nodeid 별로 데이터 저장 중...")
        save_grouped_data(data, 'nodeid', "sources/nodeid_splits")

    print_step("데이터 시각화 중...")
    visualize_data(data)

    total_time = time.time() - start_time
    print_step(f"총 소요 시간: {total_time:.2f}초")

if __name__ == "__main__":
    main()