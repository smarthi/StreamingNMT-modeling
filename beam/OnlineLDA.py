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
        parser.add_argument('--secret_number',
                            default='42')
        parser.add_argument('--input',
                              dest='input',
                              help='Input folder to process.')
        parser.add_argument('--output',
                              dest='output',
                              required=True,
                              help='Output folder to write results to.')
        parser.add_argument('--models',
                              dest='models',
                              help='Input folder to read model parameters.')
        parser.add_argument('--batchsize',
                              dest='batchsize',
                              help='Batch size for processing')

class LdaFn(beam.DoFn):

    def __init__(self):
        param = RuntimeValueProvider.get_value('secret_number', int, default_value=0)

        # The number of documents to analyze each iteration
        batchsize = 64
        # The total number of documents in Wikipedia
        self.D = 3.3e2
        # The number of topics
        self.K = 100

        self.documentstoanalyze = int(self.D/batchsize)

        # Our vocabulary
        vocab = open('../onlineldavb/dictnostops.txt', 'rt').readlines()
        self.W = len(vocab)

        # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
        self.old_alpha = OnlineLDA(vocab, self.K, self.D, 1./self.K, 1./self.K, 1024., 0.7)

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
      articles | beam.ParDo(LdaFn()) | "Write" >> WriteToText("test.txt")

if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  run()
