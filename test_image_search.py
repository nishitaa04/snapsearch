from PIL import Image
from search import search_by_image

query_path = "flower_bouquet.jpg"

results = search_by_image(query_path)
for name, score in results:
    print(f"{score:.3f}  {name}")

Image.open(query_path).show()
Image.open("data/Images/" + results[1][0]).show()