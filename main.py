import requests
import pymongo
import datetime
import uuid
from bson.json_util import dumps
from random import sample as random_sample


POLITICIANS = ["BarackObama", "realDonaldTrump", "HillaryClinton", "SpeakerPelosi"]
NETWORKS = ["NPR", "MSNBC", "CNNPolitics", "BostonGlobe", "nytimes", "FoxNews", "foxandfriends"]
ORGANIZATIONS = ["TheDemocrats", "GOP"]
CELEBS_POLITICAL = ["seanhannity"]
CELEBS = []
ALL_TWITTERS = POLITICIANS + NETWORKS + ORGANIZATIONS + CELEBS_POLITICAL + CELEBS


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
    followers_url = "https://api.twitter.com/1.1/followers/ids.json"

    @staticmethod
    def fetch_database():
        client = pymongo.MongoClient('mongodb://deltatrainer:legengerry@18.217.164.108:27017/admin?retryWrites=false',
                                     serverSelectionTimeoutMS=10)
        db = client["admin"]["17835"]
        return db

    @staticmethod
    def put_database_objects(objects: [dict]):
        """
        store objects in the 17835 bucket of the database
        :param objects:
        :return: None
        """
        assert type(objects) == list
        db = TwitterScraper.fetch_database()
        for obj in objects:
            existing_obj = db.find_one({"id": obj["id"]})
            if existing_obj is None:
                db.insert_one(obj)
            elif existing_obj["17835"]["type"] == "user":
                db.update_one({"id": obj["id"]}, {"$set": obj}, upsert=False)
            else:
                print("skipping database insertion of repeat tweet (id: {})".format(obj["id"]))

    @staticmethod
    def update_database_objects(objects: [dict]):
        """
        update objects in the database
        :param objects: ([dict]) objects to update
        :return None:
        """
        assert type(objects) == list
        db = TwitterScraper.fetch_database()
        for obj in objects:
            db.update_one({"id": obj["id"]}, {"$set": obj}, upsert=False)

    @staticmethod
    def fetch_database_objects(query: dict, fields: [str] = None):
        """
        fetch objects from the database matching given query
        :param query: (dict) the query
        :param fields: [str] optional, the fields in the objects to fetch (defaults to fetching all fields)
        :return objects: [dict] list of objects
        """
        fields = None if fields is None else {field: {"$exists": True} for field in fields}
        db = TwitterScraper.fetch_database()
        objects = list(db.find(query, fields))
        print("fetched {} objects from database".format(len(objects)))
        return objects

    @staticmethod
    def delete_objects_in_database():
        db = TwitterScraper.fetch_database()
        db.delete_many({})

    @staticmethod
    def scrape_tweets():
        """
        general tweet scraper - scrape by content
        :return tweets: [dict] the tweet objects
        """
        search_params = {"query": 'Biden "vote"', "maxResults": "100",
                         "fromDate": "<202003021200>", "toDate": "<202003022400>"}
        search_id = uuid.uuid4().hex
        search_resp = requests.get(TwitterScraper.tweet_url, params=search_params,
                                   headers=TwitterScraper.search_header)
        tweets = search_resp.json()["statuses"]

        TwitterScraper.process_objects(tweets, search_params, search_id, object_type="tweet")
        TwitterScraper.put_database_objects(tweets)
        return tweets

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
            metadata = {"date_scraped": datetime.datetime.now(), "query": query, "search_id": search_id,
                        "type": object_type}
            obj["17835"] = metadata

    @staticmethod
    def update_tweet_follow_status(tweet_id: int, follows_screen_name: str):
        db = TwitterScraper.fetch_database()
        db.update_one({"id": tweet_id}, {"$set": {"17835.follows": follows_screen_name}})

    @staticmethod
    def fetch_user_tweets(screen_name):
        """
        fetch tweets from a specific user
        :param screen_name: (str) the twitter handle
        :return objects: ([dict]) the tweet objects
        """
        objects = TwitterScraper.fetch_database_objects(query={"user.screen_name": screen_name, "17835.type": "tweet"})
        return objects

    @staticmethod
    def fetch_all_tweets(group_by=None):
        """
        fetch all tweets in the database
        :param group_by: (str) the field to group the tweets by
            (if grouped, returns tweets as dict mapping field to list of tweets in that group)
        :return objects/objects_map: ([dict]/{group: [dict]) all tweet objects (grouped if group_by is not None)
        """
        objects = TwitterScraper.fetch_database_objects(query={"17835.type": "tweet"})
        if group_by is None:
            return objects
        else:
            if group_by == "follows":
                group_func = lambda obj: obj["17835"].get("follows", None)
            elif group_by == "screen_name":
                group_func = lambda obj: obj["screen_name"]
            else:
                raise ValueError("invalid group by: {}".format(group_by))

            object_map = {}
            for obj in objects:
                key = group_func(obj)
                if key is None:
                    continue
                else:
                    object_map[key] = object_map.get(key, []) + [obj]
            return object_map

    @staticmethod
    def scrape_user_timeline(screen_name: str=None, user_id: int=None):
        """
        scrape a user's n most recent tweets (max 200)
        can also scrape multiple batches of 200 in a row, requires extra dev
        saves them in the database
        :param screen_name: (str) twitter handle of user to scrape
        :param user_id: (int) the id of the twitter user (must be passed if screen name is not)
        :return tweets: [obj] the tweet objects
        """
        search_params = {"count": 200, "exclude_replies": True,
                         "include_rts": True, "tweet_mode": "extended"}
        if screen_name is None:
            assert user_id is not None
            search_params["user_id"] = user_id
        else:
            assert user_id is None
            search_params["screen_name"] = screen_name

        search_id = uuid.uuid4().hex
        search_resp = requests.get(TwitterScraper.timeline_url, params=search_params,
                                   headers=TwitterScraper.search_header)
        tweets = search_resp.json()
        if "error" in tweets:
            # raise Exception(tweets["error"])
            print("Error: unable to scrape tweets from {}".format(screen_name if screen_name is not None else user_id))
            return []

        TwitterScraper.process_objects(tweets, search_params, search_id, object_type="tweet")
        TwitterScraper.put_database_objects(tweets)
        return tweets

    @staticmethod
    def scrape_users_timelines(screen_names: [str] = None, user_ids: [int] = None):
        """
        scrape the timelines of multiple users and return
        :param screen_names: ([str]) screen names of timelines to scrape
        :param user_ids: ([int]) user ids of timelines to scrape
        :return tweets: ([dict]) list of tweet objects
        """
        tweets = []
        if screen_names is None:
            assert user_ids is not None
            for user_id in user_ids:
                tweets += TwitterScraper.scrape_user_timeline(user_id=user_id)
        else:
            assert user_ids is None
            for screen_name in screen_names:
                tweets += TwitterScraper.scrape_user_timeline(screen_name=screen_name)
        return tweets

    @staticmethod
    def scrape_user_profiles(screen_names: [str]):
        """
        scrape the user profiles of the screen names specified then save them in the database
        :param screen_names: ([str]) a list of handles of the users to scrape
        :return profiles: [dict] the user profile objects
        """
        search_params = {"screen_name": screen_names}
        search_id = uuid.uuid4().hex
        search_resp = requests.get(TwitterScraper.user_url, params=search_params,
                                   headers=TwitterScraper.search_header)
        profiles = search_resp.json()

        TwitterScraper.process_objects(profiles, search_params, search_id, object_type="user")
        TwitterScraper.put_database_objects(profiles)
        return profiles

    @staticmethod
    def scrape_user_followers(screen_name: str):
        """
        get the user ids of accounts that follow screen_name
        :param screen_name: (str) screen name of account whos followers we want
        :return follower_ids: ([int]) user ids of followers of screen_name
        """
        user = TwitterScraper.scrape_user_profiles([screen_name])[0]

        search_params = {"screen_name": screen_name, "count": 5000}
        search_id = uuid.uuid4().hex
        search_resp = requests.get(TwitterScraper.followers_url, params=search_params,
                                   headers=TwitterScraper.search_header)

        follower_ids = search_resp.json()["ids"]
        user["17835"]["follower_ids"] = follower_ids

        TwitterScraper.put_database_objects([user])
        return follower_ids

    @staticmethod
    def scrape_users_followers(screen_names: [str]):
        """
        get the user ids of accounts that follow any of screen_names
        :param screen_names: ([str]) screen names of accounts whos followers we want
        :return follower_ids: ([int]) user ids of followers of screen_names
        """
        follower_ids = []
        for screen_name in screen_names:
            follower_ids += TwitterScraper.scrape_user_followers(screen_name)
        return follower_ids

    @staticmethod
    def scrape_user_followers_timelines(screen_name: str, n_followers: int):
        """
        get the follower ids of user then get those followers' timelines
        :param screen_name: (str) screen name of the user whos followers we want
        :param n_followers: (int) number of followers to scrape
        :return tweets: ([dict]) the aggregated tweets of all followers of screen_name
        """
        follower_ids = random_sample(TwitterScraper.scrape_user_followers(screen_name=screen_name), n_followers)
        tweets = []
        for follower_id in follower_ids:
            new_tweets = TwitterScraper.scrape_user_timeline(user_id=follower_id)
            for tweet in new_tweets:
                tweet["17835"]["follows"] = screen_name
            TwitterScraper.update_database_objects(new_tweets)
            tweets += new_tweets
        return tweets

    @staticmethod
    def scrape_users_followers_timelines(screen_names: [str], n_followers=500):
        """
        for each screen name in screen_names, get the follower ids of user then get those followers' timelines
        sorts the tweets by the screen_name whos follwoers they belong to
        :param screen_names: ([str]) screen names of the user whos followers we want
        :param n_followers: (int) number of followers to scrape
        :return tweets: ({str: [dict]}) maps screen_name to the aggregated tweets of all followers of screen_name
        """
        tweets = {}
        for screen_name in screen_names:
            tweets[screen_name] = TwitterScraper.scrape_user_followers_timelines(screen_name=screen_name, n_followers=n_followers)
        return tweets


if __name__ == '__main__':
    # tweets = TwitterScraper.scrape_users_followers_timelines(NETWORKS, n_followers=10)
    tweets = TwitterScraper.fetch_all_tweets(group_by="follows")
    # tweets = TwitterScraper.fetch_all_tweets()
    # TwitterScraper.scrape_users_timelines(ALL_TWITTERS)
    # objects = TwitterScraper.fetch_user_tweets("seanhannity")
    print()
