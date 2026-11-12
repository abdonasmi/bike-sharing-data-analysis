import pandas as pd
from pathlib import Path #pathlib is a module in Python that provides an object-oriented interface for working with file system paths.
#It allows you to manipulate and interact with file paths in a more intuitive way compared to traditional string manipulation methods.

# Get the project root directory
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def load_data():
    """Load the hourly bike sharing dataset."""
    file_path = DATA_DIR / "hour.csv"
    df = pd.read_csv(file_path)
    return df


def get_basic_info(df):
    """Return basic information about the dataset."""
    return {
        "shape": df.shape,
        "missing_values": df.isnull().sum(),
        "duplicates": df.duplicated().sum(),
        "data_types": df.dtypes
    }


def rentals_by_season(df):
    """Calculate total rentals for each season."""
    return df.groupby("season")["cnt"].sum()


def rentals_by_hour(df):
    """Calculate average rentals for each hour."""
    return df.groupby("hr")["cnt"].mean()


def rentals_by_weekday(df):
    """Calculate average rentals for each weekday."""
    return df.groupby("weekday")["cnt"].mean()


def rentals_by_weather(df):
    """Calculate average rentals for each weather condition."""
    return df.groupby("weathersit")["cnt"].mean()


def rentals_by_month(df):
    """Calculate average rentals for each month."""
    return df.groupby("mnth")["cnt"].mean()


def user_type_totals(df):
    """Calculate total casual and registered rentals."""
    return df[["casual", "registered"]].sum()


def correlation_matrix(df):
    """Calculate correlation between numerical variables."""
    columns = [
        "temp", "atemp", "hum",
        "windspeed", "casual", "registered", "cnt"
    ]

    return df[columns].corr()


def get_business_insights(df):
    """Extract important bike sharing insights."""

    hourly = rentals_by_hour(df)
    seasonal = df.groupby("season")["cnt"].mean()
    weather = rentals_by_weather(df)
    users = user_type_totals(df)

    return {
        "peak_hour": hourly.idxmax(),
        "best_season": seasonal.idxmax(),
        "best_weather": weather.idxmax(),
        "most_active_user": users.idxmax()
    }
