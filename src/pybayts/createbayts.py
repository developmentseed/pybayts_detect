import click

@click.command()
@click.option('tsL', default=None)
@click.option('pdfsdL', default=None, help='list of msdl object(s) describing the modulation of the sd of F and NF sd(F),sd(NF),mean(NF) (e.g. 2,2,-4)')
@click.option('bwf', default=None, help='block weighting function to truncate the NF probability; Default=c(0.1,0.9); (c(0,1) = no truncation)' )
def createbayts(tsL, pdfsdL, bwf):
    pass




if __name__ == 'main':
    createbayts()
