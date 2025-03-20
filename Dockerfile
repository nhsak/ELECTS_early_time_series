FROM pytorch/pytorch

# copy source code into the container
COPY . .

# dependencies
RUN pip install -r requirements.txt
