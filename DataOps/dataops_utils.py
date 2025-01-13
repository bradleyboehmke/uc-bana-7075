import json

import requests
from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi


def ingest_page_video_ids(response: requests.models.Response) -> list:
    """Ingest YouTube video IDs from a Youtube channel's individual page.
    The API will only return the first 50 video so we will need to iterate
    through several "pages" to get all the video IDs for a given channel.

    Args:
        response (requests.models.Response): Content of an API response.

    Returns:
        list: Basic video information for a given channel's API request. This
        information includes the channel ID (channel_id) along with each video's
        ID (video_id), publish date & time (datetime), and title (title).
    """

    page_video_id_list = []

    for raw_item in json.loads(response.text)["items"]:
        # only execute for youtube videos
        if raw_item["id"]["kind"] != "youtube#video":
            continue

        video_record = {}
        video_record["channel_id"] = raw_item["snippet"]["channelId"]
        video_record["video_id"] = raw_item["id"]["videoId"]
        video_record["datetime"] = raw_item["snippet"]["publishedAt"]
        video_record["title"] = raw_item["snippet"]["title"]

        page_video_id_list.append(video_record)

    return page_video_id_list


def ingest_channel_video_ids(api_key, channel_id, page_token=None):
    """Ingest all YouTube video IDs for a given Youtube channel.

    Args:
        api_key (str): Your Youtube API key.
        channel_id (str): Youtube channel ID.
        page_token (str, optional): Each API request will only return a max of 50
        results. To get all results, you must go to the next 'page'. page_token
        allows you to pull results from a single page. Setting to the defaults of
        None will start at the first page and then iterate through all available
        pages.

    Returns:
        list: Basic video information for a given channel's API request. This
        information includes the channel ID (channel_id) along with each video's
        ID (video_id), publish date & time (datetime), and title (title).
    """
    # base URL
    url = "https://www.googleapis.com/youtube/v3/search"

    # intialize list to store video data
    channel_video_id_list = []

    # extract video data across multiple search result pages
    while page_token != 0:
        # define parameters for API call
        params = {
            "key": api_key,
            "channelId": channel_id,
            "part": ["snippet", "id"],
            "order": "date",
            "maxResults": 50,
            "pageToken": page_token,
        }
        # make get request
        response = requests.get(url, params=params)

        # append video records to list
        channel_video_id_list += ingest_page_video_ids(response)

        try:
            # grab next page token
            page_token = json.loads(response.text)["nextPageToken"]
        except:
            # if no next page token kill while loop
            page_token = 0

    return channel_video_id_list


def ingest_video_stats(video_id_data, api_key):
    """Ingest basic statistics (i.e. number of views, likes & comments) for
    Youtube videos.

    Args:
        video_id_data (list): List of video ID data from `ingest_channel_video_ids()`
        api_key (_type_): Your Youtube API key.

    Returns:
        list: Same basic video information as ingest_channel_video_ids() but enhanced
        with the count of each video's views, likes, and comments.
    """
    num_iterations = len(video_id_data)
    for index, video in tqdm(
        enumerate(video_id_data),
        total=num_iterations,
        bar_format="[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}",
    ):
        url = f"https://www.googleapis.com/youtube/v3/videos?id={video['video_id']}&key={api_key}&part=statistics"
        response = requests.get(url)
        stats = json.loads(response.text)["items"][0]["statistics"]

        video_id_data[index]["views"] = stats["viewCount"]
        video_id_data[index]["likes"] = stats["likeCount"]
        video_id_data[index]["comments"] = stats["commentCount"]

    return video_id_data


def ingest_video_transcript(video_data):
    num_iterations = len(video_data)
    for index, video in tqdm(
        enumerate(video_data),
        total=num_iterations,
        bar_format="[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}",
    ):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video["video_id"])
            text_list = [transcript[i]["text"] for i in range(len(transcript))]
            transcript_text = " ".join(text_list)
        # if not available set as n/a
        except:
            transcript_text = None

        video_data[index]["transcript"] = transcript_text

    return video_data
