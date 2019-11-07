# Start

To start the docker container (from the cloned repository) :

```
docker run -d -p 32676:8888 -v "$PWD/data/":"/home/jovyan/work" --name=randgg-visualisation jupyter/randgg-visualisation
```
