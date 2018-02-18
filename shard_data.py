import os
import io
import HTMLParser

data_path = 'path/to/aggressive_dedup.json'
save_path = 'path/to/sharded_data_folder'

# generator function to read large json file
def parse(path):
  g = io.open(path,'r', encoding='utf-8')
  for l in g:
    yield eval(l)

gen = parse(data_path)
parser = HTMLParser.HTMLParser()

num_shards = 1000
reviews_per_shard = 82830

for i in range(num_shards):
    # each shard is saved in a numbered subfolder
    shard_folder = os.path.join(save_path, str(i))
    os.makedirs(shard_folder)

    with open(os.path.join(shard_folder,'input.txt'),'a') as f:
        # extract the review data from each json element
        for j in range(reviews_per_shard):
            element = gen.next()
            review = parser.unescape(element['reviewText']) # get rid of html escape characters
            review = review.encode('utf-8')
            f.write(review + '\n') # each review delimited by a newline
