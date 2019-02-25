from __future__ import absolute_import

import logging
import glob
import numpy

import apache_beam as beam
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.value_provider import RuntimeValueProvider

from onlineldavb import onlineldavb
from onlineldavb.onlineldavb import OnlineLDA


class LdaOptions(PipelineOptions):
    @classmethod
    def _add_argparse_args(cls, parser):
        parser.add_argument('--input',
                              dest='input',
                              help='Input folder to process.')
        parser.add_argument('--batchsize',
                              dest='batchsize',
                              default=10,
                              type=int,
                              help='Batch size for processing')
        parser.add_argument('--kappa',
                              dest='kappa',
                              default=0.7,
                              type=float,
                              help='Learning rate: exponential decay rate')
        parser.add_argument('--tau0',
                              dest='tau0',
                              default=1024,
                              type=int,
                              help='A (positive) learning parameter that downweights early iterations')
        parser.add_argument('--K',
                              dest='K',
                              default=100,
                              type=int,
                              help='Number of topics')

class LdaFn(beam.DoFn):

    def __init__(self, K, tau0, kappa):
        # The number of documents to analyze each iteration
        batchsize = RuntimeValueProvider.get_value('batchsize', int, default_value=64)

        # The total number of documents in Wikipedia
        self.D = 330

        # Our vocabulary
        vocab = open('../onlineldavb/dictnostops.txt', 'rt').readlines()
        self.W = len(vocab)


        # Initialize the algorithm with alpha=1/K, eta=1/K,
        self.old_alpha = OnlineLDA(vocab, K, self.D, 1./K, 1./K,
                                   tau0, kappa)

        self.iteration = 0

    def process(self, docset):
        # Give them to online LDA
        (gamma, bound) = self.old_alpha.update_lambda_docs(docset)
        # Compute an estimate of held-out perplexity
        (wordids, wordcts) = onlineldavb.parse_doc_list(docset, self.old_alpha._vocab)
        perwordbound = bound * len(docset) / (self.D * sum(map(sum, wordcts)))
        print("%d:  rho_t = %f,  held-out perplexity estimate = %f"
              % (self.iteration, self.old_alpha._rhot, numpy.exp(-perwordbound)))

        # Save lambda, the parameters to the variational distributions
        # over topics, and gamma, the parameters to the variational
        # distributions over topic weights for the articles analyzed in
        # the last iteration.
        if (self.iteration % 10 == 0):
            numpy.savetxt('lambda-%d.dat' % self.iteration, self.old_alpha._lambda)
            numpy.savetxt('gamma-%d.dat' % self.iteration, gamma)

        self.iteration = self.iteration + 1

def load_text(file_name):
    with open(file_name, 'r') as file:
        return file.read()

def run(argv=None):

  pipeline_options = PipelineOptions(argv)
  lda_options = pipeline_options.view_as(LdaOptions)
  pipeline_options.view_as(SetupOptions).save_main_session = True
  pipeline_options.view_as(StandardOptions).streaming = True

  with beam.Pipeline(options=pipeline_options) as p:
      articles = (p | "Read Articles" >> beam.Create(glob.glob(lda_options.input + '*.txt')))
      articles = articles | beam.Map(load_text)
      articles = articles | "Batch elements" >> beam.BatchElements(4, 4)
      articles | beam.ParDo(LdaFn(lda_options.K, lda_options.tau0, lda_options.kappa)) | "Write" >> WriteToText("test.txt")

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
