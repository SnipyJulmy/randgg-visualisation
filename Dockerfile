FROM jupyter/datascience-notebook:1386e2046833

COPY requirements.txt /tmp/
RUN conda install --yes --file /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
