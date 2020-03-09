import requests
import pymongo
import datetime
import uuid


class TwitterScraper:
    access_token = "AAAAAAAAAAAAAAAAAAAAADmpCwEAAAAArVTe5zTHz2ookNDQIlGImi9Fdqw%3D4GuHEFgH7ZQuSM3d4flg8eTTlafRVaTdUBrUTOJFdZMnnDQ6ji"
    client_key = "65bULXQXhB9DD9MWtiWmuj12Y"
    client_secret = "fpJFhiGd2iYQjauxCggUZYMY7mmmBRJzTJ7KQfk6pSIAYCoLSn"

    search_header = {'Authorization': 'Bearer {}'.format(access_token)}
    search_url = "https://api.twitter.com/1.1/tweets/search/30day/Test.json"

    @staticmethod
    def scrape(query):

        # search_params = {
        #     'q': query,
        #     'result_type': 'popular',
        #     # 'from_date': year,
        #     'count': 100
        # }

        search_params = {'query': 'Biden "vote"', 'maxResults': '100', 'fromDate': '202003021215', 'toDate': '202003022300'}

        search_id = uuid.uuid4().hex

        search_resp = requests.get(TwitterScraper.search_url, params = search_params, headers = TwitterScraper.search_header)
        tweets = search_resp.json()["results"]

        print(tweets[0])

        TwitterScraper.process_tweets(tweets, search_params, search_id)
        TwitterScraper.put_tweets_in_database(tweets)

    @staticmethod
    def process_tweets(tweets, query, search_id):
        for tweet in tweets:
            metadata = {"date_scraped": datetime.datetime.now(), "query": query, "search_id": search_id}
            tweet["17835"] = metadata

    @staticmethod
    def put_tweets_in_database(tweets):
        client = pymongo.MongoClient('mongodb://deltatrainer:legengerry@18.217.164.108:27017/admin?retryWrites=false', serverSelectionTimeoutMS=10)
        db = client["admin"]["17835"]
        db.insert_many(tweets)


if __name__ == '__main__':
    TwitterScraper.scrape(None)