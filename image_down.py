from icrawler.builtin import BingImageCrawler

def download_dog_images(num_images=100):
    print(f"📥 Downloading {num_images} dog images...")


    crawler = BingImageCrawler(storage={"root_dir": "images/dog"})


    crawler.crawl(
        keyword="dog",
        max_num=num_images,
        min_size=(200, 200),  
        file_idx_offset="auto"
    )

    print("✅ Download complete! Check the 'images/dog' folder.")

download_dog_images(100)
