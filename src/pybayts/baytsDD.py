import click
import createbayts

@click.command()
@click.option('tsL', default=None)
@click.option('pdfsdL', default=None, help='list of msdl object(s) describing the modulation of the sd of F and NF sd(F),sd(NF),mean(NF) (e.g. 2,2,-4)')
@click.option('distNFL', default=None, help='list of "distNF" object(s) describing the mean and sd of the NF distribution in case no data driven way to derive the NF distribution is wanted')
@click.option('formulaL', default=None, help='list of formula for the regression model')
@click.option('order', default=3, help='order of the harmonic term')
@click.option('start_history', default=None, help='start of input time series. Start date of history period used to model the seasonality and derive F and NF PDFs')
@click.option('end_history', default=None, help='Start date of history period used to model the seasonality and derive F and NF PDFs. Default=NULL (start of input time series)')
@click.option('start', default=None, help='Start date of monitoring period. Default=NULL (start of input time series)')
@click.option('end', default=None, help='End date of monitoring period.')
@click.option('bwf', default=None, help='block weighting function to truncate the NF probability; Default=c(0.1,0.9); (c(0,1) = no truncation)' )
@click.option('chi', default=0.5, help='Threshold of percent change at which the change is confirmed')
@click.option('PNFmin', default=0.5, help='reshold of pNF above which the first observation is flagged')
@click.option('residuals', default=True, help='if output are time series of deseasonanlized residuals')
def baytsDD(tsL, pdfsdL, distNFL, formulaL, order, start_history, end_history, start, end, bwf, chi, PNFmin, residuals):
    """
    docstring
    """
    pass




if __name__ == 'main':
    baytsDD()
