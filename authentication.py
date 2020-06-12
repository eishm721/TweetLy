# FILENAME: authentication.py

# After creating twitter developer account, replace the follow fields
# below with your credentials

CONSUMER_KEY = ###
CONSUMER_SECRET_KEY = ###
ACCESS_TOKEN = ###
ACCESS_SECRET_TOKEN = ###


# sample users for testing algorithms in diff categories

# Sampling criteria - famous (large number of tweets), have similar-minded people and difference (ex: left/right wing),
# many different disciplines (politics, music, business, etc...)
USERS = dict(politicians={"barackobama", "realdonaldtrump", "berniesanders", "speakerryan"},
              musicians={"selenagomez", "kendricklamar", "danielcaesar", "drake"},
              entrepreneurs={"billgates", "elonmusk", "jeffbezos"},
             sports={"djokernole", "kingjames", "tombrady", "cristiano"},
             actors={"leodicaprio", "vancityreynolds", "srbachchan", "priyankachopra"})
