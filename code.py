from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerFeedbackAnalysis").getOrCreate()

# Sample customer feedback data
data = [
    ("I love the service!", 1),
    ("The product quality is terrible.", 0),
    ("Amazing experience, will come back again!", 1),
    ("Worst customer service ever.", 0),
    ("Highly recommend this to my friends.", 1),
    ("Not worth the money.", 0)
]

columns = ["feedback", "label"]

# Create DataFrame
df = spark.createDataFrame(data, columns)

# Tokenize the feedback
tokenizer = Tokenizer(inputCol="feedback", outputCol="words")
# Remove stop words
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
# Term Frequency
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures")
# Inverse Document Frequency
idf = IDF(inputCol="rawFeatures", outputCol="features")

# Logistic Regression Model
lr = LogisticRegression(maxIter=10, regParam=0.001)

# Create Pipeline
pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

# Fit the model
model = pipeline.fit(df)

# Test the model with the same data
predictions = model.transform(df)
predictions.select("feedback", "probability", "prediction").show()

spark.stop()
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder.appName("TravelPricing").getOrCreate()

# Sample data (day, month, year, bookings, price)
data = [
    (1, 1, 2022, 150, 200),
    (2, 1, 2022, 160, 210),
    (3, 1, 2022, 170, 220),
    (4, 1, 2022, 180, 230),
    (5, 1, 2022, 190, 240)
]

columns = ["day", "month", "year", "bookings", "price"]

# Create DataFrame
df = spark.createDataFrame(data, columns)

# Assemble features
assembler = VectorAssembler(inputCols=["day", "month", "year", "bookings"], outputCol="features")
df = assembler.transform(df)

# Linear Regression model for pricing prediction
lr = LinearRegression(featuresCol="features", labelCol="price")
model = lr.fit(df)

# Test data
test_data = [
    (6, 1, 2022, 200),
    (7, 1, 2022, 210)
]

test_columns = ["day", "month", "year", "bookings"]
test_df = spark.createDataFrame(test_data, test_columns)
test_df = assembler.transform(test_df)

# Predict prices
predictions = model.transform(test_df)
predictions.select("features", "prediction").show()

spark.stop()
