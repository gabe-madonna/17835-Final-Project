import requests
import pymongo
import datetime
import uuid
from bson.json_util import dumps
from random import sample as random_sample
import numpy as np
import threading, queue
import time


BIASES = {"patribotics": -38, "Bipartisanism": -26, "fwdprogressives": -25, "HuffPost": -22, "MSNBC": -19,  "washingtonpost": -10, "CNN": -7, "propublica": -5,  "NPR": -5, "PBS": -5, "nytimes": -4, "ABC": 0, "business": 1, "CBSNews": 4, "Forbes": 5, "thehill": 10, "weeklystandard": 18, "TheTimesNUSA": 20, "amconmag": 27, "FoxNews": 27, "OANN": 28, "realDailyWire": 28, "BreitbartNews": 34, "newswarz": 38}
BIASES_IDS = {842398676714676228: -38, 487600344: -26, 1093677320810811392: -25, 14511951: -22, 2836421: -19, 2467791: -10, 759251: -7, 14606079: -5, 5392522: -5, 12133382: -5, 807095: -4, 28785486: 0, 34713362: 1, 15012486: 4, 91478624: 5, 1917731: 10, 17546958: 18, 1238468534969270273: 20, 35511525: 27, 1367531: 27, 1209936918: 28, 4081106480: 28, 457984599: 34, 1156943065426341888: 38}


POLITICIANS = ["BarackObama", "realDonaldTrump", "HillaryClinton", "SpeakerPelosi"]
NETWORKS = list(BIASES.keys())
ORGANIZATIONS = ["TheDemocrats", "GOP"]
CELEBS_POLITICAL = ["seanhannity"]
CELEBS = []
ALL_TWITTERS = POLITICIANS + NETWORKS + ORGANIZATIONS + CELEBS_POLITICAL + CELEBS


def print_json(object):
    print(dumps(object, indent=3))


def thread_run(func, arg_list:list):
    def worker_func(q, f, out_list):
        while True:
            try: (i, f_args) = q.get()
            except queue.Empty: return
            out_list[i] = f(**f_args)
            q.task_done()

    assert type(arg_list) == list
    out_list = [0] * len(arg_list)

    q = queue.Queue()
    for i, args in enumerate(arg_list):
        q.put_nowait((i, args))

    for _ in range(4):
        threading.Thread(target=worker_func, daemon=True, args=(q, func, out_list)).start()
    q.join()
    return out_list


def parallel_worker_func(f, f_args, i, out_list):
        out_list[i] = f(**f_args)


def parallel_run(target_f, arg_list):
    from multiprocessing import Manager, Pool
    import copy
    import multiprocessing

    assert type(arg_list) == list

    manager = Manager()
    out_list = manager.list([0] * len(arg_list))

    pool_size = multiprocessing.cpu_count() * 2
    pool = Pool(max(1, pool_size))

    print("running multiprocess on {} workers".format(pool_size))
    results = [pool.apply_async(parallel_worker_func, (target_f, copy.deepcopy(args), i, out_list)) for i, args in enumerate(arg_list)]
    pool.close()
    pool.join()

    for r in results:
        if isinstance(r._value, BaseException):
            raise r._value

    return list(out_list)


def merge_lists(lists):
    return sum(lists, [])


def gen_freq_dict(objects, object_field_func=lambda object: object):
    import numpy as np
    return dict(zip(*np.unique(list(map(object_field_func, objects)), return_counts=True)))


class TwitterScraper:
    access_token = "AAAAAAAAAAAAAAAAAAAAADmpCwEAAAAArVTe5zTHz2ookNDQIlGImi9Fdqw%3D4GuHEFgH7ZQuSM3d4flg8eTTlafRVaTdUBrUTOJFdZMnnDQ6ji"
    client_key = "65bULXQXhB9DD9MWtiWmuj12Y"
    client_secret = "fpJFhiGd2iYQjauxCggUZYMY7mmmBRJzTJ7KQfk6pSIAYCoLSn"

    search_header = {'Authorization': 'Bearer {}'.format(access_token)}
    tweet_url = "https://api.twitter.com/1.1/tweets/search/30day/Test.json"
    user_url = "https://api.twitter.com/1.1/users/lookup.json"
    timeline_url = "https://api.twitter.com/1.1/statuses/user_timeline.json"
    followers_url = "https://api.twitter.com/1.1/followers/ids.json"
    friendships_url = "https://api.twitter.com/1.1/friends/list.json"
    connection_url = "https://api.twitter.com/1.1/friendships/lookup.json"

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
        fields = None if fields is None else {field: True for field in fields}
        db = TwitterScraper.fetch_database()
        objects = list(db.find(query, fields))
        print("fetched {} objects from database".format(len(objects)))
        return objects

    @staticmethod
    def delete_objects_in_database():
        """
        discresionary method to delete tweet objects
        :return None: None
        """
        pass
        # db = TwitterScraper.fetch_database()
        # db.delete_many({"lang": {"$exists": True, "$ne": "en"}})

    @staticmethod
    def count_tweets(by=None):
        """
        count and print the number of tweet objects in database
        :retun None: None
        """
        if by is not None:
            fields = ["17835." + by]
            freq_f = lambda tweet: tweet["17835"].get(by, 0)
        else:
            fields = ["_id"]
        tweets = TwitterScraper.fetch_database_objects(query={"17835.type": "tweet"}, fields=fields)

        print("N tweets in database")
        if by is None:
            print(len(tweets))
            freqs = {}
        else:
            freqs = gen_freq_dict(tweets, object_field_func=freq_f)
            print(freqs)
            print("Total: {}".format(len(tweets)))
        return freqs

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
            if object_type == "user":
                try:
                    metadata["friends"] = TwitterScraper.scrape_user_friends(screen_name=obj["screen_name"])
                    metadata["bias"] = TwitterScraper.calculate_bias_from_friends(friends=metadata["friends"])
                except ValueError as error:
                    print(error)
                    metadata["friends"] = []
                    metadata["bias"] = None

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
    def fetch_user(screen_name):
        """
        fetch a specific user
        :param screen_name: (str) the twitter handle
        :return objects: ([dict]) the tweet objects
        """
        objects = TwitterScraper.fetch_database_objects(query={"screen_name": screen_name, "17835.type": "user"})
        if len(objects) == 0:
            return None
        else:
            assert len(objects) == 1
            return objects[0]

    @staticmethod
    def fetch_screennames_with_bias():
        def valid_bias(bias):
            if type(bias) is list:
                return False
            elif bias is None:
                return False
            elif np.isnan(bias):
                return False
            else:
                return True
        db = TwitterScraper.fetch_database()
        resp = db.find({"17835.type": "user", "17835.bias": {"$exists": True}}, {"screen_name": True, "17835.bias": True})
        screen_names = [r["screen_name"] for r in resp if valid_bias(r["17835"]["bias"])]
        return screen_names

    @staticmethod
    def fetch_all_tweets(group_by=None, only_average_bias=False):
        """
        fetch all tweets in the database
        :param group_by: (str) the field to group the tweets by
            (if grouped, returns tweets as dict mapping field to list of tweets in that group)
        :return objects/objects_map: ([dict]/{group: [dict]) all tweet objects (grouped if group_by is not None)
        """
        if only_average_bias:
            screen_names = TwitterScraper.fetch_screennames_with_bias()
            screen_name_query = {"$in": screen_names}
        else:
            screen_name_query = {"$exists": True}
        objects = TwitterScraper.fetch_database_objects(query={"17835.type": "tweet", "user.screen_name": screen_name_query},
                                                        fields=["full_text", "user.screen_name", "17835.bias", "17835.follows"])
        if group_by is None:
            return objects
        else:
            if group_by == "follows":
                group_func = lambda obj: obj["17835"].get("follows", None)
            elif group_by == "screen_name":
                group_func = lambda obj: obj["screen_name"]
            elif group_by == "bias":
                group_func = lambda obj: obj["17835"].get("bias", None)
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
                         "include_rts": False, "tweet_mode": "extended", "lang": "en"}
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
            print("Error: unable to scrape tweets from {} ({})".format(screen_name if screen_name is not None else user_id,
                                                                       tweets["error"]))
            return []

        # tweets = list(filter(lambda tweet: tweet["lang"] == "en", tweets))
        TwitterScraper.process_objects(tweets, search_params, search_id, object_type="tweet")
        TwitterScraper.put_database_objects(tweets)
        print("scraped {} tweets from {}".format(len(tweets), screen_name if screen_name is not None else user_id))
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
        users_per_scrape = 10
        if len(screen_names) > users_per_scrape:
            batches = [screen_names[i*users_per_scrape:min((i+1)*users_per_scrape, len(screen_names))] for i in range(len(screen_names)//users_per_scrape+1)]
            profiles = merge_lists([TwitterScraper.scrape_user_profiles(batch) for batch in batches])
            return profiles
        else:
            search_params = {"screen_name": screen_names}
            search_id = uuid.uuid4().hex
            search_resp = requests.get(TwitterScraper.user_url, params=search_params,
                                       headers=TwitterScraper.search_header)
            profiles = search_resp.json()

            TwitterScraper.process_objects(profiles, search_params, search_id, object_type="user")
            TwitterScraper.put_database_objects(profiles)
            print("scraped {} user profiles".format(len(profiles)))
            return profiles

    @staticmethod
    def scrape_user_friends(screen_name: str) -> [str]:
        """
        scrape the screen_names of the people followed by screen_name

        :param screen_name:
        :return:
        """
        cursor = -1
        friends = []
        while cursor != 0:
            search_params = {"screen_name": screen_name, "count": 200, "skip_status": True,
                             "include_user_entities": False, "cursor": cursor}
            search_resp = requests.get(TwitterScraper.friendships_url, params=search_params,
                                       headers=TwitterScraper.search_header).json()
            if "errors" in search_resp:
                if search_resp["errors"][0]["message"] == "Rate limit exceeded":
                    print("rate limit exceded, waiting...")
                    time.sleep(8 * 60)
                    continue
                else:
                    raise ValueError("Friend Scrape Errors: {}".format(search_resp["errors"]))
            elif "error" in search_resp:
                raise ValueError("Friend Scrape Error: {}".format(search_resp["error"]))

            friends += [r["screen_name"] for r in search_resp["users"]]
            cursor = search_resp["next_cursor"]
        return friends

    @staticmethod
    def calculate_bias(screen_name: str) -> float:
        search_params = {"screen_name": screen_name, "user_id": list(BIASES_IDS.values())}

        while True:
            search_resp = requests.get(TwitterScraper.connection_url, params=search_params,
                                       headers=TwitterScraper.search_header).json()
            if "errors" in search_resp:
                if search_resp["errors"][0]["message"] == "Rate limit exceeded":
                    print("rate limit exceded, waiting...")
                    time.sleep(60)
                    continue
                else:
                    raise ValueError("Friend Scrape Errors: {}".format(search_resp["errors"]))
            elif "error" in search_resp:
                raise ValueError("Friend Scrape Error: {}".format(search_resp["error"]))
            break
        follows = search_resp
        bias = 0
        return bias

    @staticmethod
    def calculate_bias_from_friends(friends: [str]) -> float:
        """
        get the bias represented by a list of friend screen names by taking out the known networks from
        friends and returning their mean bias

        :param friends:
        :return:
        """
        followed_networks = set(friends).intersection(set(BIASES.keys()))
        mean_bias = np.mean([BIASES[network] for network in followed_networks])
        return mean_bias

    @staticmethod
    def scrape_users_in_tweet_base():
        db = TwitterScraper.fetch_database()
        all_resp = list(db.find({"17835.type": "tweet"}, {"user.screen_name": True}))
        all_screen_names = gen_freq_dict([r["user"]["screen_name"] for r in all_resp])
        unknown_resp = list(db.find({"17835.type": "user", "17835.bias": {"$exists": False}}, {"screen_name": True, "friends_count": True}))
        all_n_friends = {r["screen_name"]: r.get("friends_count", 99999)//200 + 1 for r in unknown_resp}

        known_resp = list(db.find({"17835.type": "user", "17835.bias": {"$exists": True}}, {"screen_name": True}))
        known_screen_names = set([s["screen_name"] for s in known_resp])

        screen_names_set = set(all_screen_names) - set(known_screen_names)

        # #TODO: remove this
        # screen_names_set = set(filter(lambda s: s not in all_n_friends, screen_names_set))

        screen_names = sorted(list(screen_names_set), key=lambda s: (all_n_friends.get(s, 1000), -all_screen_names[s]))
        users = TwitterScraper.scrape_user_profiles(screen_names)
        print("user base up to date with tweets - scraped {} users".format(len(users)))

        return users

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
                                   headers=TwitterScraper.search_header).json()

        if "errors" in search_resp:
            print("Unable to scrape followers of {} ({})".format(screen_name, search_resp["errors"][0]["message"]))
            return []
        else:
            follower_ids = search_resp["ids"]
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
        def scrape_user_followers_helper(follower_id):
            new_tweets = TwitterScraper.scrape_user_timeline(user_id=follower_id)
            for tweet in new_tweets:
                tweet["17835"]["follows"] = screen_name
                tweet["17835"]["bias"] = BIASES.get(screen_name, None)
            TwitterScraper.update_database_objects(new_tweets)
            print("scraped {} tweets from {}".format(len(new_tweets), follower_id))
            return new_tweets

        print("Scraping followers of {}".format(screen_name))
        follower_ids_raw = TwitterScraper.scrape_user_followers(screen_name=screen_name)
        follower_ids = random_sample(follower_ids_raw, min(n_followers, len(follower_ids_raw)))
        arg_list = [{"follower_id": follower_id} for follower_id in follower_ids]
        tweets = merge_lists(thread_run(scrape_user_followers_helper, arg_list=arg_list))
        print("found {} tweets".format(len(tweets)))
        return tweets

        # print("Scraping followers of {}".format(screen_name))
        # follower_ids = random_sample(TwitterScraper.scrape_user_followers(screen_name=screen_name), n_followers)
        # tweets = []
        # for follower_id in follower_ids:
        #     new_tweets = TwitterScraper.scrape_user_timeline(user_id=follower_id)
        #     for tweet in new_tweets:
        #         tweet["17835"]["follows"] = screen_name
        #         tweet["17835"]["bias"] = BIASES.get(screen_name, None)
        #     TwitterScraper.update_database_objects(new_tweets)
        #     tweets += new_tweets
        # print("found {} tweets".format(len(tweets)))
        # return tweets

    @staticmethod
    def scrape_users_followers_timelines(screen_names: [str], n_followers=500):
        """
        for each screen name in screen_names, get the follower ids of user then get those followers' timelines
        sorts the tweets by the screen_name whos follwoers they belong to
        :param screen_names: ([str]) screen names of the user whos followers we want
        :param n_followers: (int) number of followers to scrape
        :return tweets: ({str: [dict]}) maps screen_name to the aggregated tweets of all followers of screen_name
        """
        follower_counts = TwitterScraper.count_tweets(by="follows")
        screen_names = sorted(screen_names, key=lambda screen_name: follower_counts.get(screen_name, 0))
        arg_list = [{"screen_name": screen_name, "n_followers": n_followers} for screen_name in screen_names]
        tweets = parallel_run(TwitterScraper.scrape_user_followers_timelines, arg_list=arg_list)
        tweets = merge_lists(tweets)
        # tweets = {}
        # for screen_name in screen_names:
        #     tweets[screen_name] = TwitterScraper.scrape_user_followers_timelines(screen_name=screen_name, n_followers=n_followers)
        return tweets

    @staticmethod
    def fix_tweet_biases():
        db = TwitterScraper.fetch_database()
        updated = 0
        for user in db.find({"17835.type": "user", "17835.bias": {"$exists": True}}, {"screen_name": True, "17835.bias": True}):
            if user["17835"]["bias"] == []:
                continue
            screen_name, bias = user["screen_name"], user["17835"]["bias"]
            output = db.update_many({"17835.type": "tweet", "user.screen_name": screen_name}, {"$set": {"17835.bias": bias}})
            updated += output.modified_count
        print("tweets updated: {}".format(updated))


if __name__ == '__main__':
    # TwitterScraper.scrape_users_in_tweet_base()
    # tweets = TwitterScraper.scrape_users_followers_timelines(NETWORKS, n_followers=5000)
    # tweets = TwitterScraper.fetch_all_tweets(group_by="bias", only_average_bias=True)
    tweets = TwitterScraper.fetch_all_tweets(only_average_bias=True)
    print("average tweet bias: {}".format(np.mean([tweet["17835"]["bias"] for tweet in tweets])))
    # tweets = TwitterScraper.fetch_all_tweets(only_average_bias=False)
    # objects = TwitterScraper.fetch_user_tweets("seanhannity")
    # TwitterScraper.fix_tweet_biases()
    # TwitterScraper.count_tweets(by="bias")
    print()
