FROM tensorflow/tensorflow:latest

RUN apt-get -yqq update
RUN apt-get install -yqq openssh-client openssh-server net-tools sshpass
RUN apt-get install openmpi-bin libopenmpi-dev -y
RUN pip install keras
RUN pip install mpi4py

RUN echo 'root:zhifeng' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#MaxStartups 10:30:60/MaxStartups 100/' /etc/ssh/sshd_config


ADD models /root/models/
WORKDIR  /root/models/

EXPOSE 22 57023

CMD ["/bin/bash", "./start.sh"]

