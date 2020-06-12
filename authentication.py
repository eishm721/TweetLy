# FILENAME: authentication.py

# After creating twitter developer account, replace the follow fields
# below with your credentials

CONSUMER_KEY = "Uf0jBxGSJkj1ZBhNfvcWe3mcm"
CONSUMER_SECRET_KEY = "7eHYztOGQw0GwZD0JFgcZJxzgGYP7MWhPhrojXxwvFQhZ8eoVJ"
ACCESS_TOKEN = "1256014032240365575-2AAQI52lNZxio5vjLqjVfoplsiJY8u"
ACCESS_SECRET_TOKEN = "MHCY79wY5tWhn6N23RSBLSzLpyigr54pG11bCNCimYYUr"


# sample users for testing algorithms in diff categories

# Sampling criteria - famous (large number of tweets), have similar-minded people and difference (ex: left/right wing),
# many different disciplines (politics, music, business, etc...)
USERS = dict(politicians={"barackobama", "realdonaldtrump", "berniesanders", "speakerryan"},
              musicians={"selenagomez", "kendricklamar", "danielcaesar", "drake"},
              entrepreneurs={"billgates", "elonmusk", "jeffbezos"},
             sports={"djokernole", "kingjames", "tombrady", "cristiano"},
             actors={"leodicaprio", "vancityreynolds", "srbachchan", "priyankachopra"})