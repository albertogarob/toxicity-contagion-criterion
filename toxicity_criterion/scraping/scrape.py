import argparse
import json
import time

import requests
from loadID import load_existing_ids


def main():
    parser = argparse.ArgumentParser(description="Reddit comment scraper")
    parser.add_argument("-r", "--subreddit", required=True, help="Subreddit to fetch comments from")
    parser.add_argument("-a", "--after", required=False, help="Reddit 'after' token to start pagination from")
    parser.add_argument("-d", "--delay", required=False, help="Delay between requests in seconds")

    args = parser.parse_args()

    SUBREDDIT = args.subreddit
    TARGET_COMMENTS = 5000
    OUTPUT = f"mount/{SUBREDDIT}_dataset.jsonl"
    after = args.after
    request_delay = int(args.delay) if args.delay else 2

    existing_ids = load_existing_ids(OUTPUT)

    comments = []

    print(f"Fetching comments from r/{SUBREDDIT} ...")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    while len(comments) < TARGET_COMMENTS:
        url = f"https://www.reddit.com/r/{SUBREDDIT}/comments.json?"
        if after:
            url += f"&after={after}"

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                print(f"Error: Status code {response.status_code}")
                break

            data = response.json()

            if "data" in data and "children" in data["data"]:
                children = data["data"]["children"]

                if len(children) == 0:
                    print("No more data available")
                    break

                for post in children:
                    c = post["data"]

                    if "body" not in c:
                        continue

                    parent_id = c.get("parent_id", "")

                    # Determine type
                    if parent_id.startswith("t3"):
                        ctype = "comment"
                    elif parent_id.startswith("t1"):
                        ctype = "reply"
                    else:
                        ctype = "unknown"

                    comment_id = c.get("id")
                    if comment_id in existing_ids:
                        continue

                    comments.append(
                        {
                            "id": c.get("id"),
                            "text": c.get("body", ""),
                            "author": c.get("author", "[deleted]"),
                            "type": ctype,
                            "parent_id": parent_id,
                        }
                    )

                print(f"Fetched {len(comments)} comments so far...")

                after = data["data"].get("after")
                if not after:
                    print("Reached end of available data")
                    break
            else:
                print("Unexpected response format")
                break

        except Exception as e:
            print(f"Error: {e}")
            break

        time.sleep(request_delay)

    # Save output
    if comments:
        with open(OUTPUT, "a", encoding="utf-8") as f:
            for item in comments:
                f.write(json.dumps(item) + "\n")

        print(f"\n✓ Saved {len(comments)} comments to: {OUTPUT}")
    else:
        print("\n✗ No comments were fetched")


if __name__ == "__main__":
    main()
