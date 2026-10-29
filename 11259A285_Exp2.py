
import pandas as pd
import numpy as np

# Common Data for all programs
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael',
                      'Matthew', 'Laura', 'Kevin', 'Jonas'],
             'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
             'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
             'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# 1. Create and display DataFrame
df = pd.DataFrame(exam_data, index=labels)
print(df)

# 2. Display basic information summary
print("Summary of the basic information about this DataFrame and its data:")
print(df.info())

# 3. Get first 3 rows
print("First three rows of the data frame:")
print(df.iloc[:3])

# 4. Select 'name' and 'score' columns
print("Select specific columns:")
print(df[['name', 'score']])

# 5. Select rows where attempts > 2
print("Number of attempts in the examination is greater than 2:")
print(df[df['attempts'] > 2])

# 6. Count rows and columns
total_rows = len(df.axes[0])
total_cols = len(df.axes[1])
print("Number of Rows: "+str(total_rows))
print("Number of Columns: "+str(total_cols))

# 7. Select rows where score is missing (NaN)
print("Rows where score is missing:")
print(df[df['score'].isnull()])

# 8. Append new row 'k' and delete it
print("Original rows:")
print(df)
print("\nAppend a new row:")
df.loc['k'] = ['Suresh', 15.5, 1, 'yes']
print("Print all records after insert a new record:")
print(df)
print("\nDelete the new row and display the original rows:")
df = df.drop('k')
print(df)

# 9. Sort by 'name' descending, then 'score' ascending
print("Original rows:")
print(df)
df_sorted = df.sort_values(by=['name', 'score'], ascending=[False, True])
print("Sort the data frame first by 'name' in descending order, then by 'score' in ascending order:")
print(df_sorted)

# 10. Change name 'James' to 'Suresh'
print("Original rows:")
print(df)
print("\nChange the name 'James' to 'Shyam':")
df['name'] = df['name'].replace('James', 'Shyam')
print(df)