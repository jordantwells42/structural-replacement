FROM ubuntu:20.04 as builder

ENV TZ=US/Central
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /dependencies

RUN apt-get update && \
    apt-get install -y \
        git build-essential cmake vim wget curl autotools-dev \
        python3-dev python3-pip clang libc++-dev libc++abi-dev


COPY requirements.txt .

RUN pip3 install -r requirements.txt

RUN git clone https://pose:foldtree@graylab.jhu.edu/download/PyRosetta4/git/release/PyRosetta4.MinSizeRel.python38.linux.release.git/ Pyrosetta && \
    cd Pyrosetta/setup && \
    python3 setup.py install


# ENV PYTHONPATH=/apmab:$PYTHONPATH

# instatiate with docker run -it -v $"$PWD":/root -w /root mab-engineering

ENTRYPOINT [ "bash" ]