1. Φορτώστε το docker image στο σύστημά σας την εντολή
docker load -i myapp.tar

2. Για να εκτελέσετε την εφαρμογή, εκτελέστε αυτή την εντολή
docker run -d --name python-container -p 8501:8501 datamining-app

2. Ανοίξτε το ακόλουθο url: http://localhost:8501/

 Η εφαρμογή θα αρχίσει να τρέχει
