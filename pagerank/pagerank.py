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
    print(f"PageRank Results from Sampling (n = {SAMPLES})")

    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")

    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

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
    first_page_prob = {}
    num_chain = 6
    parent_pages = {}
    parent_pages_probability = {}
    page_sample ={}

    # initialization of the very first sample


    for page in corpus:
        page_sample[page] = []
        parent_pages[page] = set()
        first_page_prob[page] = 1/ len(corpus)


    pr_value = {}
    pr_value_corr = {}

    for k in range(SAMPLES):

        current_page = random_page(corpus, first_page_prob)  # generating random page by prob for first sample
        # generat chain current-kid (parent-current)
        for i in range(num_chain):
            # surfing on current page and jump to kid page by current_page_prob
            current_page_prob = transition_model(corpus, current_page, damping_factor)  # surfing current page
            kid_page = random_page(corpus, current_page_prob)   # link to kid page from
            parent_pages[kid_page].add(current_page)  # generating current page as parent to kid page
            parent_pages_probability = current_page_prob
            current_page = kid_page

        for page in parent_pages_probability:
            page_sample[page].append(parent_pages_probability[page])

    for page in corpus:
        pr_value[page] = numpy.average(page_sample[page])

    for page in pr_value:
        pr_value_corr[page] = (1 - damping_factor) / len(corpus)
        for p_page in parent_pages[page]:
                if p_page != page:
                    num_links = len(corpus[p_page])
                    pr_value_corr[page] = float(pr_value_corr[page] + damping_factor * (pr_value[p_page] / num_links))


    return pr_value_corr

def iterate_pagerank(corpus, damping_factor):

    pr_value0 ={}
    pr_value ={}
    source_pages = {}
    num_links = {}
    for page in corpus:
        source_pages[page] = set()
        num_links[page] = len(corpus[page])

    for page in corpus:
        for link in corpus[page]:
            source_pages[link].add(page)

    # initialization of start point for pr_value[page]
    num_linked_pages = {}
    total_linked_pages=0
    for page in source_pages:
        num_linked_pages[page] = len(source_pages[page])
        total_linked_pages = total_linked_pages + len(source_pages[page])

    # pr_delta record all the changes of pr values  per page during iteration
    pr_delta = {}
    for page in corpus:
        pr_value0[page]= num_linked_pages[page]/total_linked_pages
        pr_delta[page] = 0

    flag = True
    delta = 0

    while flag:

        weight = []
        population = []
        for page in corpus:
            population.append(page)
            weight.append(100 * pr_value0[page])

        surf_page = random.choices(population, weight, k=1)
        page_prob = transition_model(corpus, surf_page[0], damping_factor)

        # change of pr_value0
        for page in corpus:
            pr_delta[page] = pr_value0[surf_page[0]]*page_prob[page]+pr_delta[page]
            delta = delta + pr_delta[page]

        # implement changes in pr_value
        pr_delta_inter = {}
        normal = 0
        for page in corpus:
            pr_delta_inter[page] = pr_delta[page]/(1+delta)
            pr_value[page] = (pr_value0[page]+pr_delta_inter[page])
            normal = pr_value[page] + normal
        pr_value_i = {}
        for page in corpus:
            pr_value_i[page] = pr_value[page]/normal

        # new pr value according to algorithom
        for page in corpus:
            pr_value[page] = 0

            for linked_page in source_pages[page]:
                pr_value[page] =pr_value_i[linked_page]/num_links[linked_page]+pr_value[page]

            pr_value[page] = (1 - damping_factor) / len(corpus)+ damping_factor*(pr_value[page])

        # check the change
        check = 0
        for page in corpus:
            check = check + abs(pr_value[page] - pr_value0[page])

        if check < 0.001:
            flag = False
        else:
            for page in corpus:
                pr_value0[page] = pr_value[page]

    return pr_value


if __name__ == "__main__":
    main()
