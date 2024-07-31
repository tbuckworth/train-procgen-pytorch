from simfin.names import REVENUE, NET_INCOME

import simfin as sf

if __name__ == "__main__":
    # Set your API-key for downloading data.
    # Replace YOUR_API_KEY with your actual API-key.
    sf.set_api_key('8764b2f4-f668-479f-8e97-0dd22746af62')

    # Set the local directory where data-files are stored.
    # The dir will be created if it does not already exist.
    sf.set_data_dir('~/simfin_data/')

    # Load the annual Income Statements for all companies in the US.
    # The data is automatically downloaded if you don't have it already.
    df = sf.load_income(variant='annual', market='us')

    # Print all Revenue and Net Income for Microsoft (ticker MSFT).
    print(df.loc['MSFT', [REVENUE, NET_INCOME]])



    print("something")
