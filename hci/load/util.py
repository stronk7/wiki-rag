#  Copyright (c) 2025, Moodle HQ - Research
#  SPDX-License-Identifier: BSD-3-Clause

""" Util functions to proceed to load and parse the mediawiki pages"""
import re

import requests
import logging
import time
import random
import uuid

from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_mediawiki_pages_list(mediawiki_url: str, namespaces: list[int], user_agent: str, chunk: int = 500) -> list[dict]:
    """
    Get the list of pages from the mediawiki API.

    :param mediawiki_url: The url of the mediawiki site.
    :param namespaces: The list of namespaces to get the pages from.
    :param user_agent: The user agent to use in the requests.
    :param chunk: The number of pages to get per request.
    :return: The list of pages.
    """

    api_url = f"{mediawiki_url}/api.php"
    
    # Get an estimation about the number of pages to load.
    headers = {
        "User-Agent": user_agent
    }
    params = {
        'action': 'query',
        'format': 'json',
        'meta': 'siteinfo',
        'siprop': 'statistics',
    }
    
    session = requests.Session()
    
    result = session.get(url=api_url, params=params, headers=headers)
    articles = result.json()["query"]["statistics"]["articles"]
    max_chunks = (articles * len(namespaces) // chunk) + 1
    
    pages = []
    next_page = None
    for namespace in namespaces:
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'allpages',
            'apnamespace': namespace,
            "apfilterredir": "nonredirects",
            'aplimit': chunk,
            "apdir": "ascending",
        }
        with tqdm(total=max_chunks, desc=f"Loading namespace {namespace}") as pbar:
            while True:
                if next_page:
                    params["apcontinue"] = next_page
    
                result = session.get(url=api_url, params=params, headers=headers)
                data = result.json()
                pages.extend(data["query"]["allpages"])
    
                if "continue" in data:
                    next_page = data["continue"]["apcontinue"]
                else:
                    break
    
                time.sleep(random.uniform(2, 5))  # We aren't in a hurry (it's only a few requests).
                pbar.update(1)
    
    return pages

def get_mediawiki_parsed_pages(mediawiki_url: str, pages: list[dict], user_agent: str) -> list[dict]:
    """
    Parse the pages and split them into sections.

    :param mediawiki_url: The url of the mediawiki site.   
    :param pages: The list of pages to parse.
    :param user_agent: The user agent to use in the requests.
    :return: The list of parsed pages.
    """
    parsed_pages = []
    # TODO: Remove this line, it's only for testing purposes (100 last pages).
    pages = pages[-60:-40]
    for page in tqdm(pages, desc=f"Processing pages", unit="page"):
        time.sleep(random.uniform(2, 3))  # We aren't in a hurry (it's only a few requests).
        try:
            sections, categories, internal_links, external_links, language_links = parse_page(
                mediawiki_url, page["pageid"], user_agent)  # Parse pages and sections.
            tidy_sections_text(mediawiki_url, sections, categories, internal_links, external_links,
                               language_links)  # Tidy up contents and links.
            calculate_relationships(sections)  # Calculate all the relationships between sections.
            parsed_pages.append({
                "id": page["pageid"],
                "title": page["title"],
                "sections": sections,
                "categories": categories,
                "internal_links": internal_links,
                "external_links": external_links,
                "language_links": language_links,
            })
        except Exception as e:
            logger.error(f"  Error processing page \"{page["title"]}\": {e}")
        finally:
            continue

    # Now that all the pages and their sections are in memory, we can convert any wiki link to a "relation" to the target section.
    # (that will improve the context organisation later, providing one more way to navigate the information).
    convert_internal_links(parsed_pages)  # Convert internal wiki links to point to existing UUIDs.
    
    return parsed_pages

def parse_page(mediawiki_url: str, page_id: int, user_agent: str) -> list:
    api_url = f"{mediawiki_url}/api.php"
    headers = {
        "User-Agent": user_agent
    }
    params = {
        "action": "parse",
        "format": "json",
        "pageid": page_id,
        "prop": "title|revid|wikitext|sections|categories|links|langlinks|externallinks|subtitle",
    }

    session = requests.Session()
    result = session.get(url=api_url, params=params, headers=headers)

    id = result.json()["parse"]["pageid"]
    revision_id = result.json()["parse"]["revid"]
    title = result.json()["parse"]["title"]
    categories = [cat["*"] for cat in result.json()["parse"]["categories"]]
    internal_links = [link["*"] for link in result.json()["parse"]["links"] if "exists" in link]
    external_links = result.json()["parse"]["externallinks"]
    language_links = [f"{lang["lang"]}:{lang["*"]}" for lang in result.json()["parse"]["langlinks"]]

    # Based on the URL and the page id, create a stable document identifier for the whole page.
    doc_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{mediawiki_url}/{id}")

    # And, based on the doc_id and the revision id (we don't need the real content, the revision is unique),
    # create the document hash for future checks.
    doc_hash = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}-{revision_id}".encode())

    # Let's process the sections, adding the corresponding text to them.
    sections_info = result.json()["parse"]["sections"]
    sections = []
    # If there aren't sections or the fist section doesn't start at 0, we need to add a section with the text from the beginning of the page to it.
    if not sections_info or sections_info[0]["byteoffset"] != 0 :
        section_zero = {
            "anchor": "",
            "line": title,
            "byteoffset": 0,
            "index": 0,
            "level": 1,
        }
        sections_info.insert(0, section_zero)

    text_end = len(result.json()["parse"]["wikitext"]["*"])
    for section in sections_info[::-1]: # Going backwards to be able to split based on byteoffset.
        section_anchor = section["anchor"]
        section_title = section["line"]
        section_source = f"{mediawiki_url}/{str.replace(title, ' ', '_')}#{section_anchor}".rstrip("/#?") # Very basic built.
        section_byteoffset = section["byteoffset"]
        # Now extract from section_byteoffset to text_end
        section_text = result.json()["parse"]["wikitext"]["*"][section_byteoffset:text_end]
        section_index = int(section["index"]) if section["index"] else 0
        section_level = int(section["level"])
        section_id = uuid.uuid5(uuid.NAMESPACE_OID, f"{doc_id}-{section_index}".encode())
        section = {
            "id": section_id,
            "anchor": section_anchor,
            "title": section_title,
            "source": section_source,
            "text": section_text,
            "all_links": [],
            "wiki_links": [],
            "index": section_index,
            "level": section_level,
            "doc_id": doc_id,
            "doc_title": title,
            "doc_hash": doc_hash,
            "parent": None,
            "children": [],
            "previous": [],
            "next": [],
            "relations": [],
        }
        text_end = section_byteoffset - 1
        # TODO: If the section is too big, we should split it here in smaller parts before continuing. Some simple text splitting.
        #       Alternatively, we can just crop and done, but we'll lose some context.
        sections.append(section)

    sections.reverse() # And back again to the original order.
    return [sections, categories, internal_links, external_links, language_links]

# Now we need to edit every section text to apply various text transformations.
def tidy_sections_text(mediawiki_url, sections, categories, internal_links, external_links, language_links):
    for section in sections:
        # Remove all the categories information from the text.
        for cat in categories:
            section["text"] = section["text"].replace(f"[[Category:{cat}]]", "")
        # Remove all the language links from the text.
        for lang in language_links:
            section["text"] = section["text"].replace(f"[[{lang}]]", "")
        # Replace all the internal links from the text to their description, if available, else by their title.
        for link in internal_links:
            # Look for all the internal links in the text matching the pattern [[link(\|description)?]].
            # For those having description, we'll replace the matching text with the description. If the description
            # is not available, we'll replace the matching text with the link.
            pattern = re.compile(rf"\[\[{re.escape(link)}(\|(.+?))?\]\]" )
            matches = pattern.findall(section["text"])
            for match in matches:
                if match[1]:
                    section["text"] = re.sub(pattern, match[1], section["text"])
                else:
                    section["text"] = re.sub(pattern, link, section["text"])
                section["all_links"].append(f"{mediawiki_url}/{str.replace(link, ' ', '_')}")  # Very basic built.
                section["wiki_links"].append(link) # Also keep the internal wiki links, we'll find their UUIDs later.
        # Replace all the external links from the text to their description, if available, else by their url.
        for link in external_links:
            # Look for all the external links in the text matching the pattern [url description].
            # For those having description, we'll replace the matching text with the description. If the description
            # is not available, we'll replace the matching text with the url.
            pattern = re.compile(rf"\[{re.escape(link)}(\s+(.+?))?\]")
            matches = pattern.findall(section["text"])
            for match in matches:
                if match[1]:
                    section["text"] = re.sub(pattern, match[1], section["text"])
                else:
                    section["text"] = re.sub(pattern, link, section["text"])
                section["all_links"].append(link)
        # Remove images and files. TODO: Analyse if we want them back, at least listing them like the links.
        pattern = re.compile(rf"\[\[(File|Image):.+?\]\]")
        section["text"] = re.sub(pattern, "", section["text"])
        # Remove all templates in the text.
        pattern = re.compile(r"\{\{.+?\}\}", re.DOTALL | re.MULTILINE)
        section["text"] = re.sub(pattern, "", section["text"])
        # Remove all section headings in the text. We'll be adding them later in a tree-ish structure.
        pattern = re.compile(r"==+.+?==+")
        section["text"] = re.sub(pattern, "", section["text"])
        # Final touches, whitespace trim the text and remove double line feeds.
        section["text"] = re.sub(r"\n{2,}", "\n", section["text"].strip())
        
def calculate_relationships(sections):

    parent_candidates = {}
    for section in sections:
        # Reset any previous parent/child relationship information.
        section["parent"] = None
        section["children"] = []
        # Remove from parent_candidates all the levels higher or equal to the current section level,
        # the current section will become a parent candidate for the next sections.
        for level in list(parent_candidates.keys()):
            if level >= section["level"]:
                parent_candidates.pop(level)

        # If there is any remaining parent candidate, the last one is our parent.
        if parent_candidates:
            parent = parent_candidates[list(parent_candidates.keys())[-1]]
            section["parent"] = parent["id"]
            parent["children"].append(section["id"])

        # Add  current element as a parent candidate.
        parent_candidates[section["level"]] = section

    # With all parent/child relationships in place, we can now calculate the previous/next relationships.
    for section in sections:
        # Given a section having children elements, we need to update
        # every children section with the list of previous and next
        # sibling sections.
        if section["children"]:
            for child in section["children"]:
                # Get the target section from the sections list, by id.
                child_to_update = [s for s in sections if s["id"] == child][0]
                # All the elements in the children list are siblings, let's get
                # those before (previous) and after (next) the current child.
                child_index = section["children"].index(child)
                if child_index > 0:
                    child_to_update["previous"] = section["children"][:child_index]  # From the beginning to the previous.
                if child_index < len(section["children"]) - 1:
                    child_to_update["next"] = section["children"][child_index + 1:] # From the next to the end.

# Let's iterate over all pages and their sections and, if they contain internal (wiki) links, try to find to which section they point.
def convert_internal_links(pages):
    for page in tqdm(pages, desc=f"Converting wiki links", unit="sections"):
        sections = page["sections"]
        for section in sections:
            for link in section["wiki_links"]:
                # Look for the section with the title matching the link.
                # If the link has an anchor, we have to concatenate the doc_title and the title, else, only the title.
                if "#" in link:
                    target = [s for s in sections if f"{s["doc_title"]}#{s["title"]}" == link]
                else:
                    target = [s for s in sections if s["doc_title"] == link]
                if target:
                    section["relations"].append(target[0]["id"])