IMAGE_ID=$1
IS_GPU=$2

if [[ "$IS_GPU" == "" ]]; then
    CONTRAINER_ID=$(docker run -d $IMAGE_ID) # Run in detached container
else
    CONTRAINER_ID=$(docker run --gpus=all -d $IMAGE_ID)
fi

PATTERN="Initalized model"
WAIT_FREQ=2
TEMP_LOGFILE="/tmp/$(uuidgen)"

echo "Trying to init docker images. Streaming tail of log every $WAIT_FREQ seconds ..."
n=1
LAST_CONTENT=0
MAX_TIME=300 # 5 min
MAX_WAIT=$((MAX_TIME / WAIT_FREQ))
IS_FAILED=0

while true
do
    (docker logs -f $CONTRAINER_ID --tail 1 &> $TEMP_LOGFILE) &
    PID=$!
    sleep $WAIT_FREQ
    kill $PID &>/dev/null

    CURRENT_CONTENT=$(cat $TEMP_LOGFILE)
    if [ "$LAST_CONTENT" != "$CURRENT_CONTENT" ] ; then
        cat $TEMP_LOGFILE
        LAST_CONTENT=$CURRENT_CONTENT
    fi

    if grep -q "$PATTERN" $TEMP_LOGFILE; then
        echo "Init sucessfully. Breaking"
        break
    fi

    ALL_EXITED_CONTAINER="$(docker ps -f status=exited)"
    if grep -q $CONTRAINER_ID <<< $ALL_EXITED_CONTAINER; then
        echo "Container have exited. Looks like something went wrong. Breaking with fail ..."
        IS_FAILED=1
        break
    fi

    if [ $n -gt $MAX_WAIT ]; then
        echo "Init tooks more than 5 minutes. Breaking with fail ..."
        IS_FAILED=1
        break
    fi

    n=$(( n+1 ))
done

# Clean up
docker kill $CONTRAINER_ID
rm $TEMP_LOGFILE

if [ $IS_FAILED -eq 1 ]; then
    exit 1;
fi