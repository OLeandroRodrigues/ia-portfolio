# main.py


from google_comments_scraper.scraper import GoogleCommentScraper

if __name__ == "__main__":
    url = "https://www.google.com/search?q=loja+ofner+perdizes&oq=loja+ofner+perdizes&gs_lcrp=EgZjaHJvbWUqCggAEAAY4wIYgAQyCggAEAAY4wIYgAQyEAgBEC4YrwEYxwEYgAQYjgUyBwgCEAAY7wUyCggDEAAYgAQYogQyCggEEAAYgAQYogQyCggFEAAYgAQYogTSAQgzNDc4ajBqNKgCALACAQ&sourceid=chrome&ie=UTF-8&sei=0d-QaPLIKLCC5OUPiN2OkAo#lrd=0x94ce57f55b7f4dad:0xb1e756042b056e2d,1,,,,"
    scraper = GoogleCommentScraper(url, output_file="ofner_comentarios.txt", max_comments=50)
    scraper.run()