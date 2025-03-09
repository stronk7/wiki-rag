# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html),
and commits should be formattted using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

## [0.6.0] - 2025-03-09

### Added

- Develop: Get commit messages and changelogs under control by @stronk7 at ([1556f88](https://github.com/moodlehq/wiki-rag/commit/1556f8820f48ae465c0bc0dda00327d9bb35f3da))
- Docs: Automate CHANGELOG.md generation by @stronk7 at ([092f094](https://github.com/moodlehq/wiki-rag/commit/092f094e290c03f591a2f1857314a8a4e34d0c5b))
- Release: Automate GitHub releases on tagging by @stronk7 at ([d35ea69](https://github.com/moodlehq/wiki-rag/commit/d35ea69496975cb3225c95a604f83ad81b310997))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.5.6...v0.6.0

## [0.5.6] - 2025-03-08

### Added

- Install: Add support for all-in-one installation with docker compose by @stronk7 at ([a8a6a58](https://github.com/moodlehq/wiki-rag/commit/a8a6a585bc19295f5b68438f29f9f76511451c7a))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.5.5...v0.5.6

## [0.5.5] - 2025-03-08

### Added

- Add OCI annotations that is what ghcr.io registry uses by @stronk7 at ([e8c90f6](https://github.com/moodlehq/wiki-rag/commit/e8c90f63f5b110af61a389ecfb6d1e63afbf4e57))
- Add instructions to run it as a docker container by @stronk7 at ([1c1f5b4](https://github.com/moodlehq/wiki-rag/commit/1c1f5b46f552470f126d8c90c13a2e519d1a9608))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.5.4...v0.5.5

## [0.5.4] - 2025-03-08

### Changed

- Split Dockerfile into builder and runner by @stronk7 at ([15ccc03](https://github.com/moodlehq/wiki-rag/commit/15ccc03f4ca6640c2193062eeb26ae8d38b5eea2))
- Set PR author as assignee by @stronk7 at ([6d43dbf](https://github.com/moodlehq/wiki-rag/commit/6d43dbfe6ddee57a31b0771bd885b849b1b74dde))
- Build the docker images on push (tags & latest) by @stronk7 at ([8990608](https://github.com/moodlehq/wiki-rag/commit/89906083e92d1ffff07b636fbc0a87365afe9b80))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.5.3...v0.5.4

## [0.5.3] - 2025-03-06

### Changed

- Bump dependencies to current + update some link by @stronk7 at ([04d9a2d](https://github.com/moodlehq/wiki-rag/commit/04d9a2dbfe81914926d6e62a83100ca74497b112))
- Attempt to create the "data" directory if not present by @stronk7 at ([9eb3b5f](https://github.com/moodlehq/wiki-rag/commit/9eb3b5fe14af6aabcb1419748122ff3877103453))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.5.2...v0.5.3

## [0.5.2] - 2025-03-05

### Added

- Add preliminary env_template file by @stronk7 at ([b0db040](https://github.com/moodlehq/wiki-rag/commit/b0db0406a8affbcf0e12cbc2fb8aa985e8a8ffc9))

### Changed

- Fix some project URLs, docs and defaults by @stronk7 at ([a328dc8](https://github.com/moodlehq/wiki-rag/commit/a328dc84a7346b9d827e740d6b650469c6b33c73))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.5.1...v0.5.2

## [0.5.1] - 2025-03-04

### Changed

- Ignore the volumes dir (created by milvus) by @stronk7 at ([021fe8f](https://github.com/moodlehq/wiki-rag/commit/021fe8fe6916fff1b143337d901823c25d7f24c5))
- Initial documentation details by @stronk7 at ([60ff816](https://github.com/moodlehq/wiki-rag/commit/60ff81640b1d601c1749c8a19b17f0bc7040c5e0))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.5.0...v0.5.1

## [0.5.0] - 2025-03-04

### Added

- Add docker support by @stronk7 at ([0ecb2e9](https://github.com/moodlehq/wiki-rag/commit/0ecb2e91b59024cf6ca528e1d148246691dac448))

### Changed

- Few improvements towards better working in a container by @stronk7 at ([6e3960d](https://github.com/moodlehq/wiki-rag/commit/6e3960d2ca89ab7e69981b2a75c934e53644c9f8))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.4.1...v0.5.0

## [0.4.1] - 2025-03-03

### Added

- Add support for exclusions by @stronk7 at ([454d2bc](https://github.com/moodlehq/wiki-rag/commit/454d2bc244dfd933589239413589c73437f0f4d9))

### Changed

- Small improvements to auth, better logging and caching by @stronk7 at ([9a867fc](https://github.com/moodlehq/wiki-rag/commit/9a867fc70eaef4924da3ce1a214bbfd41cea98dd))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.4.0...v0.4.1

## [0.4.0] - 2025-03-02

### Added

- Add authentication to all end-points by @stronk7 at ([b8f07f0](https://github.com/moodlehq/wiki-rag/commit/b8f07f0c1718c37abb3e5e68231d6872b3f6df85))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.3.4...v0.4.0

## [0.3.4] - 2025-03-01

### Changed

- Tidy up the local prompt fallback by @stronk7 at ([fdf4aa0](https://github.com/moodlehq/wiki-rag/commit/fdf4aa011f551c9126ed9d2179b6ada1880e2489))
- Few improvements to server (globals, responses, deprecations...) by @stronk7 at ([204de9c](https://github.com/moodlehq/wiki-rag/commit/204de9cdadee4b0d1db0617c1f229204889d40e3))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.3.3...v0.3.4

## [0.3.3] - 2025-02-27

### Added

- Add page_id to the index and search results for easier tracking by @stronk7 at ([dc08e2b](https://github.com/moodlehq/wiki-rag/commit/dc08e2bcd3f0c0e89efe1e316613bb41ba89dcdc))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.3.2...v0.3.3

## [0.3.2] - 2025-02-27

### Changed

- Small changes to sources generation by @stronk7 at ([5bde66b](https://github.com/moodlehq/wiki-rag/commit/5bde66b9d189088dad9b4e3a9ff66532b2548e59))
- Make all search / graph operations async by @stronk7 at ([a4ba97b](https://github.com/moodlehq/wiki-rag/commit/a4ba97b4add67b2641fff68839677171016b4516))
- Small improvements to prompt handling an others by @stronk7 at ([fdd8062](https://github.com/moodlehq/wiki-rag/commit/fdd80629059f7c0e179053cbbdc92a69b1355ade))
- Cache load_prompts_for_rag() by @stronk7 at ([202cc73](https://github.com/moodlehq/wiki-rag/commit/202cc73539587b2e3c7f8d33aa49eb4e11659f6d))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.3.1...v0.3.2

## [0.3.1] - 2025-02-24

### Added

- Add basic support to show sources by @stronk7 at ([788e3a1](https://github.com/moodlehq/wiki-rag/commit/788e3a19da81db61147853cca0ef8632833c30a6))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.3.0...v0.3.1

## [0.3.0] - 2025-02-24

### Changed

- Move package name from hci to wiki_rag by @stronk7 at ([0f30291](https://github.com/moodlehq/wiki-rag/commit/0f3029173656fd9b28a0edb149ebb9e011553045))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.2.1...v0.3.0

## [0.2.1] - 2025-02-24

### Changed

- Coding style (ruff) fixes by @stronk7 at ([bf570fa](https://github.com/moodlehq/wiki-rag/commit/bf570fab600d10e32e382d96fb3ba95aa7eebbae))
- Typing (pyright) fixes by @stronk7 at ([e101179](https://github.com/moodlehq/wiki-rag/commit/e101179d3bdb8a2a0c52daf0f3c3997f3d8328c7))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.2.0...v0.2.1

## [0.2.0] - 2025-02-22

### Added

- Add basic support for incoming chat history by @stronk7 at ([ebd4fd3](https://github.com/moodlehq/wiki-rag/commit/ebd4fd31dd33b6a7092d04497390fbb0002e4511))
- Add basic support for chat history by @stronk7 at ([88ddef7](https://github.com/moodlehq/wiki-rag/commit/88ddef7b81efb42e79fa91e64bff849aefa5e93f))

### Changed

- Small adjustments to search (system prompt, checks, ...) by @stronk7 at ([70b7a69](https://github.com/moodlehq/wiki-rag/commit/70b7a69b43c9580e723d16e8eacfc724afa80cb6))
- Whitespace fixes by @stronk7 at ([9b33cbc](https://github.com/moodlehq/wiki-rag/commit/9b33cbcb2eb1f022ee496be2de9e02d450d013d0))
- First cut of the OpenAi-compatible server by @stronk7 at ([73c3878](https://github.com/moodlehq/wiki-rag/commit/73c3878e19a07f75816c2859d648159340ecbc3b))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.1.3...v0.2.0

## [0.1.3] - 2025-02-20

### Added

- Add support for streaming answers in search CLI by @stronk7 at ([8573c90](https://github.com/moodlehq/wiki-rag/commit/8573c9078f3e967fa4024b3cf975fb9f7731c656))

### Changed

- Better support the progress when multiple namespaces are processed by @stronk7 at ([55e976e](https://github.com/moodlehq/wiki-rag/commit/55e976eaec382a2936d8dd4c62e3cd63b695d3f7))
- By default always index the last (by name) file in the data dir by @stronk7 at ([ad06c85](https://github.com/moodlehq/wiki-rag/commit/ad06c8578074febaf2420926dc5c7aa7f4533281))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.1.2...v0.1.3

## [0.1.2] - 2025-02-17

### Changed

- First cut to the search utility by @stronk7 at ([b03bf68](https://github.com/moodlehq/wiki-rag/commit/b03bf68e9d6aa1f67349bcdfcbaecd59820357ba))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.1.1...v0.1.2

## [0.1.1] - 2025-02-16

### Changed

- Completed the indexing page by @stronk7 at ([4468437](https://github.com/moodlehq/wiki-rag/commit/4468437ad394b39d07dd08e52e3a921fa3d14704))

**Full Changelog**: https://github.com/moodlehq/wiki-rag/compare/v0.1.0...v0.1.1

## [0.1.0] - 2025-02-15

### Changed

- Initial commit, implementing the loader and project basis by @stronk7 at ([c1b282a](https://github.com/moodlehq/wiki-rag/commit/c1b282acce3b24cac516df36b6105a0ba05c2945))
- Completed the load phase, results are dump to file by @stronk7 at ([a24a644](https://github.com/moodlehq/wiki-rag/commit/a24a644f40584f12ce2a4aea992066cf3713be7a))

### New Contributors 🧡:

- @stronk7 made their first contribution


[0.6.0]: https://github.com/moodlehq/wiki-rag/compare/v0.5.6..v0.6.0
[0.5.6]: https://github.com/moodlehq/wiki-rag/compare/v0.5.5..v0.5.6
[0.5.5]: https://github.com/moodlehq/wiki-rag/compare/v0.5.4..v0.5.5
[0.5.4]: https://github.com/moodlehq/wiki-rag/compare/v0.5.3..v0.5.4
[0.5.3]: https://github.com/moodlehq/wiki-rag/compare/v0.5.2..v0.5.3
[0.5.2]: https://github.com/moodlehq/wiki-rag/compare/v0.5.1..v0.5.2
[0.5.1]: https://github.com/moodlehq/wiki-rag/compare/v0.5.0..v0.5.1
[0.5.0]: https://github.com/moodlehq/wiki-rag/compare/v0.4.1..v0.5.0
[0.4.1]: https://github.com/moodlehq/wiki-rag/compare/v0.4.0..v0.4.1
[0.4.0]: https://github.com/moodlehq/wiki-rag/compare/v0.3.4..v0.4.0
[0.3.4]: https://github.com/moodlehq/wiki-rag/compare/v0.3.3..v0.3.4
[0.3.3]: https://github.com/moodlehq/wiki-rag/compare/v0.3.2..v0.3.3
[0.3.2]: https://github.com/moodlehq/wiki-rag/compare/v0.3.1..v0.3.2
[0.3.1]: https://github.com/moodlehq/wiki-rag/compare/v0.3.0..v0.3.1
[0.3.0]: https://github.com/moodlehq/wiki-rag/compare/v0.2.1..v0.3.0
[0.2.1]: https://github.com/moodlehq/wiki-rag/compare/v0.2.0..v0.2.1
[0.2.0]: https://github.com/moodlehq/wiki-rag/compare/v0.1.3..v0.2.0
[0.1.3]: https://github.com/moodlehq/wiki-rag/compare/v0.1.2..v0.1.3
[0.1.2]: https://github.com/moodlehq/wiki-rag/compare/v0.1.1..v0.1.2
[0.1.1]: https://github.com/moodlehq/wiki-rag/compare/v0.1.0..v0.1.1

<!-- generated by git-cliff -->
