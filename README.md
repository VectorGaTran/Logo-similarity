### Solution Explanation
by Trandafir Victor-Gabriel
I chose the grouping of logos problem because I have experimented in University with computer vision and machine learning and thought it would be the challenge I would be most able to solve.

I needed to parse logo images, so I looked at the first 10 domains and concluded that I needed to handle various image types (JPGs, PNGs, and SVGs). The SVG files were later converted to PNGs and all images were eventually converted to PIL for further editing and preprocessing, to handle challenges like different resolutions, distortions, and variations in design. I standardized all logos to a 128x128 size.

To extract logos, I fetched the URLs of image files written in page sources, by either looking for the words "logo" or "branding" in the image files themselves or within their divs. Some domains have "condensed" logos, but I made the program disregard those if other logo-containing files or divs were available. This approach can slow down the process, as it initially fetches more images per site, but I thought it was necessary for better accuracy.

When a site wouldn’t load (due to timeouts) or the logo couldn’t be found in the usual sources, I used the Clearbit Logo API (https://clearbit.com/blog/logo) to return images based on domains. However, Clearbit’s logo often returns condensed logos (simple pictograms), which are of better quality than favicons but still less ideal.

To handle HTTP requests, I created a function using the "requests" library and turned the domain data from the parquet file into a pandas dataframe. To bypass 403 errors, I added a custom header, as I found that the site would work when using that header.

I also experimented with Selenium and ChromeDriver to scrape dynamically and reduce reliance on Clearbit. However, the execution times were slower by around 10x, with little improvement in results. All my tests were done on the first 10 domains in the list. I documented all the variants of the code I tested in a separate folder.

Once I confirmed the correct logos were being extracted, I parallelized the logo-fetching process using threads. I also tried using PySpark for distributed processing, but it didn’t work well locally in VSCode (I had used PySpark previously in Databricks), but it can be used in the future for further optimizing by scaling the solution.

Used suggestive prints and execution time measurement for debugging.

---

### Feature Extraction

After gathering the logos, I experimented with traditional computer vision techniques like ORB (Oriented FAST and Rotated BRIEF) and HOG (Histogram of Oriented Gradients) because they are faster, more robust, and efficient than SIFT. I used a fixed size 200 features to extract relevant details from the greyscaled versions of the logos. I also incorporated the color data of the logos by computing a color histogram to help with clustering (since I believe color plays a vital role in logo identity).

---

### Deep Learning Approach

In the end, I opted for a deep learning approach using the pre-trained ResNet50 model. I chose ResNet50 over other CNNs, vision transformers, or triplet loss-based models due to its proven efficiency in general image feature extraction tasks. ResNet50 has a strong ability to capture hierarchical features and can work well with high-dimensional data like logos. It also offers a balance of computational efficiency and accuracy for the given task. Although ResNet50 is primarily designed for image classification (and not specifically for logos), it gave quicker and more satisfying results when handling more complex images and larger datasets.

I resized the images to 224x224 and normalized the pixel values using the ImageNet dataset statistics, which offered more stability and speed. I avoided models like Vision Transformers or Siamese networks in this case, as they require more training data and would be overkill for the given dataset, making them slower to train and deploy.

I stored the domain names and their corresponding logo data (converted into numpy arrays) in a dictionary. I also implemented error handling to ensure the robustness of the program.

---

### Logo Comparison and Clustering

Data was flattened and scaled.  
For clustering, I chose Agglomerative Clustering with a distance threshold (based on heuristic fine-tuning). I avoided K-means, as the problem could involve clusters containing only one domain, and I couldn’t estimate the number of clusters in advance. Based on the first 10 logos, I couldn't reliably group them myself, as I am not highly brand-aware. An alternative method I could have used was DBSCAN for density clustering, but I found hierarchical clustering more fitting for this case. In the future, I would consider implementing silhouette scores to evaluate the quality of the clusters.

---

### Results and Output

After performing the clustering, I saved the results—grouped domains—into a JSON file (though a CSV approach could do as well). This allowed me to output the final results, showing the logos grouped into clusters of similar designs.

---

### Challenges and Alternatives

I initially experimented with approaches like using favicons with BeautifulSoup and Selenium, but these solutions didn’t provide the expected results. As mentioned earlier, I faced difficulties in dynamically fetching logos with Selenium and using Clearbit’s API for logos that weren't available directly on the site. I also tried SIFT and SimCLR for feature extraction, but ultimately, the deep learning-based ResNet50 approach yielded better results for logo similarity.

The deep learning solution with ResNet50 provided the most consistent and scalable results for clustering logos.

---

### Future Approaches

In the future, I would explore more advanced techniques to improve the clustering process. I also experimented with a Siamese network for training on positive and negative pairs, but after testing, I found that the ResNet50 approach was more reliable for my use case.

For comparing logo similarities, metrics like SSIM (Structural Similarity Index), perceptual hashing, and cosine similarity to compare deep learning embeddings could be used. Additionally, silhouette scores could be incorporated to assess the quality of the generated clusters.

### Conclusion

My approach focused on robustness, speed, and accuracy in fetching logos and clustering them effectively. While I explored a range of traditional methods like ORB, HOG, and clustering algorithms, the deep learning method proved to be more effective for the dataset in question. In the future, I would explore more advanced techniques like vision transformers or Siamese networks to further improve clustering accuracy and logo similarity matching. I also plan to handle challenges like different resolutions, distortions, and variations in design for better results.

---

### Additional Notes

On my old ThinkPad laptop, the solution would to process all 4400 domains in the `logos.snappy.parquet` file in about an hour. The JSON sample in the repository contains the clustering results for half of those domains.

