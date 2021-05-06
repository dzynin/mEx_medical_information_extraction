# set base image (host OS)
#https://blog.baudson.de/blog/stop-and-remove-all-docker-containers-and-images
FROM python:3.6

# Set Working Directory
WORKDIR /opt/app

# Copy needed files from local space
COPY . .

# Download the RelationExtraction Model from the DFKI cloud
#ADD https://cloud.dfki.de/owncloud/index.php/s/e7G9deea7eRksCY/download ./pos.pt
#ADD https://cloud.dfki.de/owncloud/index.php/s/WWbnqJ6N8gQQWMD/download ./ner.pt
#ADD https://cloud.dfki.de/owncloud/index.php/s/zDH7FHNbXQXkcLx/download ./relex.pt

# Install python libraries
RUN pip install --no-cache-dir -r requierments.txt

# Expose the 5006 port to let flask use it
EXPOSE 5006

# Run the CMD as an entry point
CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0", "--port=5006"]


# docker build -t relex-flask-restapi .
# docker run -d -p 5006:5006 --name relexapi relex-flask-restapi