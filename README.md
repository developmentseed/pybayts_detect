```
docker build -t pybaytsserver .
```

```
docker run -it --rm \          
    -v $HOME/.aws:/root/.aws \
    -v "$(pwd)":/home/work/pybayts \     
    -p 8888:8888 \
    -e AWS_PROFILE=devseed \
    --ipc=host pybaytsserver
```