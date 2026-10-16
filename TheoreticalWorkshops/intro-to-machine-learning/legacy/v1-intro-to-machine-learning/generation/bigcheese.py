# Generate a pd dataset as a csv file, that is on variable vs another which is a linear combination of the other with an intercept and some noise.

import pandas as pd
import numpy as np

# Generate a dataset with 1000 samples
n = 100

# Generate a random variable x that is normally distributed
x = np.random.normal(5, 5, n)
x = np.maximum(x, 0)

# Generate a random variable y
y = 1 / 2 * x + 1 + np.random.normal(0, 1, n)

x = np.concatenate([x, np.zeros(n // 10)])
y = np.concatenate([y, np.random.rand(n // 10) * 5])

# Round the values of y to the nearest integer and make sure they are positive
y = np.round(y)
y = np.maximum(y, 0)

# Create a dataframe
df = pd.DataFrame(
    {"Units of alcohol per week": x, "Big cheese attendances per year": y}
)

# Save the dataframe to a csv file
df.to_csv(
    "./IntroToML/data/bigcheese.csv", index=False
)

# Now generate quadratic data
y = 0.1 * x**2 - 0.8 * x + 1.5 + np.random.normal(0, 0.5, n + 10)

# Round the values of y to the nearest integer and make sure they are positive
y = np.round(y)
y = np.maximum(y, 0)
y = np.minimum(y, 20)

# Create a dataframe
df = pd.DataFrame(
    {"Units of alcohol per week": x, "Big cheese attendances per year": y}
)

# Save the dataframe to a csv file
df.to_csv(
    "./IntroToML/data/bigcheese-quadratic.csv",
    index=False,
)
