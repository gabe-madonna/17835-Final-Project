import requests
import pymongo
import datetime
import uuid
from bson.json_util import dumps

# profiles we've already fetched (we have their id numbers)
PROFILES = ["BarackObama", "realDonaldTrump", "HillaryClinton",
            "TheDemocrats", "SpeakerPelosi", "NPR", "MSNBC",
            "CNNPolitics", "BostonGlobe", "nytimes", "FoxNews",
            "seanhannity", "GOP", "foxandfriends"]


def print_json(object):
    print(dumps(object, indent=3))


class TwitterScraper:
    access_token = "AAAAAAAAAAAAAAAAAAAAADmpCwEAAAAArVTe5zTHz2ookNDQIlGImi9Fdqw%3D4GuHEFgH7ZQuSM3d4flg8eTTlafRVaTdUBrUTOJFdZMnnDQ6ji"
    client_key = "65bULXQXhB9DD9MWtiWmuj12Y"
    client_secret = "fpJFhiGd2iYQjauxCggUZYMY7mmmBRJzTJ7KQfk6pSIAYCoLSn"

    search_header = {'Authorization': 'Bearer {}'.format(access_token)}
    tweet_url = "https://api.twitter.com/1.1/tweets/search/30day/Test.json"
    user_url = "https://api.twitter.com/1.1/users/lookup.json"
    timeline_url = "https://api.twitter.com/1.1/statuses/user_timeline.json"

    @staticmethod
    def scrape_tweets():
        """
        general tweet scraper - scrape by content
        :return: None
        """
        search_params = {"query": 'Biden "vote"', "maxResults": "100",
                         "fromDate": "<202003021200>", "toDate": "<202003022400>"}
        search_id = uuid.uuid4().hex
        search_resp = requests.get(TwitterScraper.tweet_url, params=search_params,
                                   headers=TwitterScraper.search_header)
        tweets = search_resp.json()["statuses"]

        TwitterScraper.process_objects(tweets, search_params, search_id, object_type="tweet")
        TwitterScraper.put_objects_in_database(tweets)

    @staticmethod
    def scrape_user_timeline(screen_name: str):
        """
        scrape a user's n most recent tweets (max 200)
        can also scrape multiple batches of 200 in a row, requires extra dev
        saves them in the database
        :param screen_name: twitter handle of user to scrape
        :return: None
        """
        search_params = {"screen_name": screen_name, "count": 200, "exclude_replies": True,
                         "include_rts": True, "tweet_mode": "extended"}
        search_id = uuid.uuid4().hex
        search_resp = requests.get(TwitterScraper.timeline_url, params=search_params,
                                   headers=TwitterScraper.search_header)
        tweets = search_resp.json()

        TwitterScraper.process_objects(tweets, search_params, search_id, object_type="tweet")
        TwitterScraper.put_objects_in_database(tweets)

    @staticmethod
    def scrape_user_profiles(screen_names: [str]):
        """
        scrape the user profiles of the screen names specified then save them in the database
        :param screen_names: ([str]) a list of handles of the users to scrape
        :return:
        """
        search_params = {"screen_name": screen_names}
        search_id = uuid.uuid4().hex
        search_resp = requests.get(TwitterScraper.user_url, params=search_params,
                                   headers=TwitterScraper.search_header)
        profiles = search_resp.json()

        TwitterScraper.process_objects(profiles, search_params, search_id, object_type="user")
        TwitterScraper.put_objects_in_database(profiles)

    @staticmethod
    def process_objects(objects: [dict], query: dict, search_id: str, object_type: str):
        """
        add metadata to the 17835 attribute of the object and put it in the database
        modifies objects in place

        :param objects: ([dict])
        :param query: ([dict]) query used to fetch obejcts from twitter
        :param search_id: (str) the id of the query/search
        :param object_type: (str) the type of the object (user, tweet, ...)
        :return: None
        """
        for obj in objects:
            metadata = {"date_scraped": datetime.datetime.now(), "query": query, "search_id": search_id, "type": object_type}
            obj["17835"] = metadata

    @staticmethod
    def put_objects_in_database(objects: [dict]):
        """
        store objects in the 17835 bucket of the database
        :param objects:
        :return: None
        """
        client = pymongo.MongoClient('mongodb://deltatrainer:legengerry@18.217.164.108:27017/admin?retryWrites=false', serverSelectionTimeoutMS=10)
        db = client["admin"]["17835"]
        db.insert_many(objects)

    @staticmethod
    def fetch_database_objects(query: dict, fields: [str]=None):
        """
        fetch objects from the database matching given query
        :param query: (dict) the query
        :param fields: [str] optional, the fields in the objects to fetch (defaults to fetching all fields)
        :return objects: [dict] list of objects
        """
        fields = None if fields is None else {field: {"$exists": True} for field in fields}
        client = pymongo.MongoClient('mongodb://deltatrainer:legengerry@18.217.164.108:27017/admin?retryWrites=false',
                                     serverSelectionTimeoutMS=10)
        db = client["admin"]["17835"]
        objects = list(db.find(query, fields))
        print("fetched {} objects from database".format(len(objects)))
        return objects

    @staticmethod
    def fetch_user_tweets(screen_name):
        """
        fetch tweets from a specific user
        :param screen_name: (str) the twitter handle
        :return objects: ([dict]) the tweet objects
        """
        objects = TwitterScraper.fetch_database_objects(query={"user.screen_name": screen_name, "17835.type": "tweet"})
        return objects


if __name__ == '__main__':
    # TwitterScraper.scrape_tweets()
    # TwitterScraper.scrape_user_timeline("BarackObama")
    # TwitterScraper.scrape_user_profiles()
    objects = TwitterScraper.fetch_user_tweets("BarackObama")
    print_json(objects[6])
