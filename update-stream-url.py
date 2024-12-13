# TODO: remove options
USE_SELENIUMWIRE = True
USE_PROXY = False
PROXY_URL = "http://brd-customer-hl_c581f9ac-zone-datacenter_proxy2:zhwtg7u2ascp@brd.superproxy.io:33335"

############################################

from urllib3.exceptions import ReadTimeoutError
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

def restart_driver(use_seleniumwire=True):
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver_kwargs = dict(
        service=Service('/usr/bin/chromedriver'),
        options=chrome_options,
    )

    if use_seleniumwire:
        from seleniumwire import webdriver

        seleniumwire_options = {}

        if USE_PROXY:
            # chrome_options.add_argument(f"--proxy-server={PROXY_URL}")
            seleniumwire_options.update({
                "proxy": {
                    "http": PROXY_URL,
                    "https": PROXY_URL
                },
            })
        driver_kwargs.update({'seleniumwire_options': seleniumwire_options})
    else:
        from selenium import webdriver

    # Initialize driver with infinite retries
    global driver
    is_driver_initialized = False
    while not is_driver_initialized:
        try:
            if 'driver' in globals():
                print("Driver quitting...")
                driver.quit()
                time.sleep(2)
                print("Done.")

            if use_seleniumwire:
                print("Initializing selniumwire driver...")
            else:
                print("Initializing selenium driver...")

            driver = webdriver.Chrome(**driver_kwargs)
            driver.command_executor._client_config.timeout = 120
            # print("Post-init timeout:", driver.command_executor._client_config.timeout)
            is_driver_initialized = True
            print("Done.")
        except ReadTimeoutError as e:
            print("Timeout error when initializing driver. Retrying...")
            print(e)
            pass

from functools import wraps
from selenium.common.exceptions import (
    InvalidSessionIdException, StaleElementReferenceException, NoSuchElementException, TimeoutException
)
from urllib3.exceptions import MaxRetryError, ReadTimeoutError

# Decorator to handle retries
def retry_on_failure(use_seleniumwire=False):
    def _retry_on_failure(func):
        @wraps(func)
        def wrapper(*args, throw_errors=False, **kwargs):
            global driver
            if throw_errors:
                func_out = func(*args, **kwargs)
            else:
                is_func_executed = False
                while not is_func_executed:
                    try:
                        func_out = func(*args, **kwargs)
                        is_func_executed = True
                        return func_out
                    except (
                        MaxRetryError,
                        ReadTimeoutError,
                        InvalidSessionIdException,
                        StaleElementReferenceException,
                        NoSuchElementException,
                        TimeoutException,
                        IndexError
                    ) as e:
                        print(f"Error {type(e)} while executing function. Restarting driver...")
                        restart_driver(use_seleniumwire)
                        pass

        return wrapper
    return _retry_on_failure

import re

@retry_on_failure(use_seleniumwire=True)
def update_stream_url():
    landing_url = "https://www.m24.ru/"
    driver.get(landing_url)
    urls = [req.url for req in driver.requests]
    matched_urls = []
    for url in urls:
        match = re.search(r'air/\d+p\.m3u8', url)
        if match is not None:
            matched_urls.append(url)

    matched_urls = set(matched_urls)
    assert len(matched_urls) == 1
    news_url = matched_urls.pop()
    return news_url

import os

if __name__ == '__main__':
    restart_driver(use_seleniumwire=True)
    news_url = update_stream_url()

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the .env file in the same directory as the script
    env_file_path = os.path.join(script_dir, ".env")

    # Write to the .env file
    with open(env_file_path, "w") as f:
        f.write(f"NEWS_URL={news_url}\n")
    
    print("NEWS_URL updated in .env")