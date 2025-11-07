import os
import json
import requests
from dotenv import load_dotenv
import pandas as pd
import time

# -
# Helper Functions
# -

def safe_get(data_dict, key_path, default=None):
    """Safely get nested dictionary or list values using dot notation."""
    keys = key_path.split('.')
    val = data_dict
    try:
        for key in keys:
            if isinstance(val, list):
                try:
                    key = int(key)
                except ValueError:
                    return default
            val = val[key]
        return val
    except (KeyError, TypeError, IndexError):
        return default

def determine_format_category(ad_creative):
    """Classify ad creative into Video, Carousel, or Static Image."""
    if not ad_creative:
        return "Unknown"
    if safe_get(ad_creative, 'asset_feed_spec.videos') or safe_get(ad_creative, 'object_story_spec.video_data.video_id'):
        return "Video/Reel"
    if safe_get(ad_creative, 'object_story_spec.link_data.child_attachments'):
        return "Carousel"
    if safe_get(ad_creative, 'image_url') or safe_get(ad_creative, 'asset_feed_spec.images') or safe_get(ad_creative, 'image_hash') or safe_get(ad_creative, 'thumbnail_url') or safe_get(ad_creative, 'object_story_spec.photo_data'):
        return "Static Image"
    return "Unknown"

def fetch_image_urls(hash_list, access_token, ad_account_id, api_version="v24.0"):
    """Query Meta API for real image URLs based on image_hashes."""
    if not hash_list:
        return {}
    print(f"Step 2A: Resolving {len(hash_list)} image hashes...")
    hash_url_map = {}
    url = f"https://graph.facebook.com/{api_version}/act_{ad_account_id}/adimages"
    params = {
        'fields': 'hash,url',
        'hashes': json.dumps(list(hash_list)),
        'access_token': access_token
    }
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        for item in res.json().get('data', []):
            h, u = item.get('hash'), item.get('url')
            if h and u:
                hash_url_map[h] = u
    except Exception as e:
        print("❌ Error fetching image URLs:", e)
    return hash_url_map

def fetch_video_urls(video_ids, access_token, api_version="v24.0"):
    """Resolve video IDs into playable source URLs."""
    if not video_ids:
        return {}
    print(f"Step 2B: Resolving {len(video_ids)} video IDs...")
    video_url_map = {}
    BASE_URL = f"https://graph.facebook.com/{api_version}/"
    for vid in video_ids:
        params = {'fields': 'source', 'access_token': access_token}
        try:
            res = requests.get(f"{BASE_URL}{vid}", params=params)
            res.raise_for_status()
            data = res.json()
            # Check specifically for the 'source' field
            source_url = data.get('source')
            if source_url:
                video_url_map[vid] = source_url
            else:
                # Log if 'source' is missing, even if the request was successful
                print(f"⚠ No 'source' field found for video {vid}. Skipping.")
        except requests.exceptions.HTTPError as e:
            # Check if it's a permission error (#10)
            if e.response.status_code == 400:
                try:
                    error_data = e.response.json().get('error', {})
                    if error_data.get('code') == 10: # (#10) Application does not have permission
                        print(f"⚠ Skipping video {vid} due to permission error (#10): {error_data.get('message', 'Permission denied')}")
                    else:
                        print(f"⚠ Skipping video {vid} due to HTTP 400: {e.response.text}")
                except json.JSONDecodeError:
                    print(f"⚠ Skipping video {vid} due to HTTP 400: {e.response.text}")
            else:
                print(f"⚠ Skipping video {vid} due to HTTP Error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            print(f"⚠ Skipping video {vid} due to unexpected error: {e}")
    print(f"✅ Successfully resolved {len(video_url_map)} video URLs out of {len(video_ids)}.")
    return video_url_map


# -
# Main Script
# -

def get_data_script():
    load_dotenv()
    ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN")
    AD_ACCOUNT_ID = os.getenv("META_AD_ACCOUNT_ID")
    API_VERSION = "v24.0"

    if not all([ACCESS_TOKEN, AD_ACCOUNT_ID]):
        print("❌ Missing Meta Ads credentials in .env file.")
        return

    # Create directories
    os.makedirs("data", exist_ok=True)

      # -
# Step 1: Fetch all ads and creatives (WITHOUT INSIGHTS initially)
# -
    FIELDS = (
        "id,name,status,"
        "campaign{id,name,objective,status,daily_budget,start_time,stop_time},"
        "adset{id,name,optimization_goal,status,daily_budget,"
        "targeting{geo_locations,countries,age_min,age_max,genders,"
        "publisher_platforms,facebook_positions,instagram_positions}},"
        "creative{title,body,image_hash,thumbnail_url,image_url,"
        "asset_feed_spec{videos{video_id},images{url}},"
        "object_story_spec{"
        "text_data{message},"
        "link_data{link,name,description,caption,picture,"
        "child_attachments{link,description,image_hash,picture,name,caption,call_to_action}},"
        "video_data{video_id,image_url,title,call_to_action{type,value}},"
        "photo_data{image_hash,url}"
        "}}"
        # Removed: ",insights.time_range({'since':'2025-10-01','until':'2025-10-28'}){spend,impressions,clicks,ctr,cpc,cpm,actions,results,cost_per_action_type,purchase_roas}"
    )

    url = f"https://graph.facebook.com/{API_VERSION}/act_{AD_ACCOUNT_ID}/ads"
    params = {
        'access_token': ACCESS_TOKEN,
        'fields': FIELDS,
        'limit': 100
    }
    ads = []
    hashes = set()
    video_ids = set()

    print("Step 1: Fetching ads...")
    page = 1
    while url:
        print(f"  - Fetching page {page}...")
        res = requests.get(url, params=params if page == 1 else {})
        res.raise_for_status()
        data = res.json()

        page_ads = data.get("data", [])
        ads.extend(page_ads)

        # Collect hashes and video IDs for later resolution
        for ad in page_ads:
            creative = ad.get("creative", {})
            ad["format_category"] = determine_format_category(creative)

            # Collect image hashes
            h = safe_get(creative, "image_hash")
            if h:
                hashes.add(h)
            for att in safe_get(creative, "object_story_spec.link_data.child_attachments", []):
                if att.get("image_hash"):
                    hashes.add(att["image_hash"])
            photo_data = safe_get(creative, "object_story_spec.photo_data")
            if photo_data and 'image_hash' in photo_data:
                hashes.add(photo_data['image_hash'])

            # Collect video IDs
            for v in safe_get(creative, "asset_feed_spec.videos", []):
                if "video_id" in v:
                    video_ids.add(v["video_id"])
            vid = safe_get(creative, "object_story_spec.video_data.video_id")
            if vid:
                video_ids.add(vid)

        url = data.get("paging", {}).get("next")
        page += 1
        time.sleep(0.1) # Small delay to be respectful to the API

    print(f"✅ Total fetched: {len(ads)} ads across {page - 1} pages, found {len(hashes)} unique hashes and {len(video_ids)} unique video IDs.")


    # -
    # Step 2: Resolve Media URLs
    # -
    hash_url_map = fetch_image_urls(hashes, ACCESS_TOKEN, AD_ACCOUNT_ID, API_VERSION)
    video_url_map = fetch_video_urls(video_ids, ACCESS_TOKEN, API_VERSION) # This will now handle errors gracefully


    # -
    # Step 3: Attach Resolved URLs into Creatives
    # -
    print("Step 3: Injecting resolved media URLs back into ad data...")
    for ad in ads:
        creative = ad.get("creative", {})

        # --- IMAGE RESOLUTION ---
        top_hash = safe_get(creative, "image_hash")
        if top_hash and top_hash in hash_url_map:
            creative["image_url"] = hash_url_map[top_hash]

        # Carousel child attachments
        children = safe_get(creative, "object_story_spec.link_data.child_attachments", [])
        for child in children:
            child_hash = child.get("image_hash")
            if child_hash and child_hash in hash_url_map:
                child["image_url"] = hash_url_map[child_hash]

        # Photo_data hash
        photo_data = safe_get(ad, 'creative.object_story_spec.photo_data')
        if photo_data and 'image_hash' in photo_data:
            photo_hash = photo_data['image_hash']
            if photo_hash and photo_hash in hash_url_map:
                photo_data['url'] = hash_url_map[photo_hash]


        # --- VIDEO RESOLUTION ---
        # 1. Asset Feed Spec Videos
        for v in safe_get(creative, "asset_feed_spec.videos", []):
            vid = v.get("video_id")
            if vid and vid in video_url_map:
                # Add the resolved source URL as a new field
                v["source_url"] = video_url_map[vid]

        # 2. Object Story Spec Video Data
        video_data = safe_get(creative, "object_story_spec.video_data")
        if video_data and "video_id" in video_data:
            vid = video_data["video_id"]
            if vid and vid in video_url_map:
                # Add the resolved source URL as a new field
                video_data["video_url"] = video_url_map[vid] # This is the field you wanted


    # -
    # Step 4: Save raw JSON locally (with resolved URLs)
    # -
    json_path = "data/dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ads, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved processed dataset (with resolved media URLs) to {json_path}")


    # -
    # Step 5: Flatten into CSV (WITH INSIGHTS)
    # -
    print("Step 5: Flattening data into CSV...")
    records = []
    for ad in ads:
        campaign = ad.get("campaign", {})
        adset = ad.get("adset", {})
        creative = ad.get("creative", {})

        # Parse insights correctly - assume single data point for this range
        ins_data = ad.get("insights", {}).get("data", []) # Get the list of insight dicts
        ins = ins_data[0] if ins_data else {} # Take the first (and likely only) entry

        # Safely extract metrics, defaulting to 0
        spend = float(ins.get("spend", 0))
        impressions = int(ins.get("impressions", 0))
        clicks = int(ins.get("clicks", 0))
        ctr = float(ins.get("ctr", 0))
        cpc = float(ins.get("cpc", 0))
        cpm = float(ins.get("cpm", 0))

        # Handle purchase_roas (list or dict)
        roas_val = 0
        roas_field = ins.get("purchase_roas")
        if isinstance(roas_field, list) and len(roas_field) > 0:
            val = roas_field[0].get("value")
            if val:
                roas_val = float(val)

        # Derive conversions and conversion rate
        actions = ins.get("actions", [])
        conversions = sum(int(a.get("value", 0)) for a in actions if a.get("action_type") in ["offsite_conversion", "purchase", "onsite_conversion.purchase"])
        conversion_rate = (conversions / clicks * 100) if clicks > 0 else 0

        # Get the resolved video URL if available
        resolved_video_url = safe_get(creative, "object_story_spec.video_data.video_url") # Use the new field name

        rec = {
            "ad_id": ad.get("id"),
            "ad_name": ad.get("name"),
            "ad_status": ad.get("status"),
            "format_category": ad.get("format_category"),
            "campaign_id": campaign.get("id"),
            "campaign_name": campaign.get("name"),
            "campaign_objective": campaign.get("objective"),
            "adset_id": adset.get("id"),
            "adset_name": adset.get("name"),
            "optimization_goal": adset.get("optimization_goal"),
            "creative_title": creative.get("title"),
            "creative_body": creative.get("body"),
            "creative_image_url": creative.get("image_url"), # This is now the resolved image URL
            "creative_video_url": resolved_video_url, # This will be the resolved video URL, or None/empty if not resolved
            "creative_thumbnail_url": creative.get("thumbnail_url"),
            "copy_text": safe_get(creative, "object_story_spec.text_data.message"),
            "link_url": safe_get(creative, "object_story_spec.link_data.link"),
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "ctr": ctr,
            "cpc": cpc,
            "cpm": cpm,
            "roas": roas_val,
            "conversions": conversions,
            "conversion_rate": conversion_rate,
        }
        records.append(rec)

    df = pd.DataFrame(records)
    csv_path = "data/meta_ads_data.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Exported {len(df)} ads to {csv_path}")


if __name__ == "__main__":
    get_data_script()