import pandas as pd
import matplotlib.pyplot as plt
import os


def load_nodeid_data(directory, nodeid, files_per_nodeid=10):
    all_data = []
    nodeid_files = [f for f in os.listdir(directory) if f.startswith(f'nodeid_{nodeid}') and f.endswith('.csv')]

    for filename in nodeid_files[:files_per_nodeid]:
        filepath = os.path.join(directory, filename)
        data = pd.read_csv(filepath, parse_dates=['datetime'])
        all_data.append(data)

    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values(by='datetime').drop_duplicates(subset=['datetime'])
        return combined_data
    else:
        raise FileNotFoundError(f"No files found for nodeid {nodeid} in {directory}")


def preprocess_data(data):
    data = data.sort_values(by='datetime')
    data = data.drop_duplicates(subset=['datetime'])
    data['speed'] = data['speed'].ffill()
    return data


def save_processed_data(data, nodeid, output_directory):
    nodeid_directory = os.path.join(output_directory, nodeid)
    if not os.path.exists(nodeid_directory):
        os.makedirs(nodeid_directory)

    # Save the processed CSV
    output_csv_path = os.path.join(nodeid_directory, f'{nodeid}.csv')
    data.to_csv(output_csv_path, index=False)
    print(f"Processed data saved to {output_csv_path}")

    # Save the data description to a text file
    description = data.describe().to_string()
    output_txt_path = os.path.join(nodeid_directory, f'{nodeid}.txt')
    with open(output_txt_path, 'w') as f:
        f.write(description)
    print(f"Data description saved to {output_txt_path}")

    return nodeid_directory


def perform_eda(data, nodeid_directory):
    print(data.describe())

    # 시간대별 평균 속도 시각화
    data.set_index('datetime', inplace=True)

    # 시간에 따른 속도 변화를 시각화합니다.
    plt.figure(figsize=(18, 8))
    plt.plot(data.index, data['speed'], linewidth=1)
    plt.title('Speed Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('Speed')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 분석 결과 이미지를 저장합니다.
    plot_filepath = os.path.join(nodeid_directory, 'speed_over_time.png')
    plt.savefig(plot_filepath)
    plt.close()
    print(f"Plot saved to {plot_filepath}")


def main():
    directory = "sources/nodeid_splits"
    output_directory = "sources/processed"
    nodeid = input("Enter nodeid: ")

    try:
        data = load_nodeid_data(directory, nodeid, files_per_nodeid=10)
        data = preprocess_data(data)
        nodeid_directory = save_processed_data(data, nodeid, output_directory)
        perform_eda(data, nodeid_directory)
    except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
    main()
