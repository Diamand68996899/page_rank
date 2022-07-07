import os
import random
import re
import sys
import numpy

DAMPING = 0.85
SAMPLES = 10000
CURR_DIR = os.getcwd()   ## delete after
print(CURR_DIR)            ## delete after

goal_dir = os.path.join(CURR_DIR, "corpus0")    ##
print(goal_dir)   ##

def main():
    ##if len(sys.argv) != 2:
    ##    sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(goal_dir)   ## sys.argv[1]  replaced
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)   # page: rank
    ## corpus = {'1.html': {'2.html'}, '2.html': {'1.html', '3.html'}, '3.html': {'4.html', '2.html'}, '4.html': {'2.html'}}
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    ###

    ##for page in sorted(ranks):
      ##  print(f"  {page}: {ranks[page]:.4f}")
    ## ranks = iterate_pagerank(corpus, DAMPING)
    ##print(f"PageRank Results from Iteration")

    for page in sorted(ranks):
        #print('____________>',page, sorted(ranks))
        print(f"  {page}: {ranks[page]:.4f}")
        #print(page)   ## to be deleted


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}
    print(pages)
    # Only include links to other pages in the corpus
    for filename in pages:  ## access value by keys, filename = 1.html
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )                    ## pages['1.html'] = set
    print(pages)
    return pages


def transition_model(corpus, page, damping_factor):
    random_page = page
    num_links = len(corpus[random_page])
    page_next_prob = {}

    if num_links > 0:
        for link in corpus:
            if link == page:
                page_next_prob[random_page] = (1 - damping_factor) / (num_links + 1)
            elif link in corpus[random_page]:
                page_next_prob[link] = (1 - damping_factor) / (num_links + 1) + damping_factor / num_links
            else:
                page_next_prob[link] = 0

    else:                            # no link on the page
        for link in corpus:
            page_next_prob[link] = 1/len(corpus)

    return page_next_prob         # output: {page: probability}


def sample_pagerank(corpus, damping_factor, n):
    #  ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    #    for page in sorted(ranks):      # sorted by key values
    #         print(f"  {page}: {ranks[page]:.4f}")

    def random_page(corpus, page_prob):
        population = []
        prob = []
        # initialization of pr_value for each page in corpus
        for page in corpus:
            prob.append(page_prob[page])
            population.append(page)

        # surfer from one random page with equal chance for each page
        random_state = random.choices(population, weights=prob,  k=1) # random_state = ['2.html']
        random_page = random_state[0]

        return random_page

    # transit state
    page_prob = {}
    num_chain = 5
    parent_pages = {}
    parent_pages_probability = {}
    current_page = ''
    page_sample ={}

    for page in corpus:
        page_sample[page] = []
        parent_pages[page] = set()

    for j in range(SAMPLES):

        for i in range(num_chain):
            if len(page_prob) == 0:    #   initialization first random page, start
                for page in corpus:
                    page_prob[page] = 1/len(corpus)
                random_current_page = random_page(corpus, page_prob)  # generating random page by prob
                current_page = random_current_page

            # next_page by calling transition_model return {next_page:prob}
            current_page_prob = transition_model(corpus, current_page, damping_factor)  # surfing current page
            kid_page = random_page(corpus, current_page_prob)   # link to kid page from
            parent_pages[kid_page].add(current_page)  # generating current page as parent to kid page
            parent_pages_probability = current_page_prob
            current_page = kid_page

        for page in parent_pages_probability:
            page_sample[page].append(parent_pages_probability[page])


    pr_value = {}
    for page in corpus:
        print(page, numpy.average(page_sample[page]))
        pr_value[page] = numpy.average(page_sample[page])
        print(page, parent_pages[page])

    # correct with algorithm
    print(pr_value)

    pr_value_corr = {}
    for page in pr_value:
        pr_value_corr[page] = (1-damping_factor)/len(corpus)
        for p_page in parent_pages[page]:
            if p_page != page:
                num_links = len(corpus[p_page])
                pr_value_corr[page] = pr_value_corr[page] + damping_factor*(pr_value[p_page]/num_links)

    print(pr_value_corr)
    return pr_value

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
