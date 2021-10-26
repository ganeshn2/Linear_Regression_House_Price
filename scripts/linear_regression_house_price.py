import pandas as pd
from helper_function_files import normality_test


def load_clean_dataframe():
    USA_house_price_df = pd.read_csv("input/USA_Housing.csv")
    USA_house_price_df = USA_house_price_df.drop(labels = 'Address', axis =1)
    USA_house_price_df.rename(columns={'Avg. Area Income': 'Income',
                                                            'Avg. Area House Age': 'House_Age',
                                                            'Avg. Area Number of Rooms': 'No_Rooms',
                                                            'Avg. Area Number of Bedrooms': 'No_Bedrooms'
                                                            })
    return USA_house_price_df


if __name__ == "__main__":
    USA_house_df = load_clean_dataframe()
    boxcox_test = normality_test(USA_house_df['Price'])[2]
