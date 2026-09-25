# Manual labeling criteria

A thread is `yes` when it discusses **code duplication**, even in a single passing remark. It does not need to end in a dedup diff. The label describes the whole thread, not the term that made it pass the pre-filter.

## `yes` categories

The `category` column of the consolidated sample stores the name in parentheses after each category, the same names the classifiers in `classify/` use. Several categories are joined with commas, in the order below.

Categories 1 and 2 can occur together, because a thread may have several patches and each one may do something different. Category 3 never occurs together with categories 1 and 2. Category 4 can occur alone or together with any other category.

The diff is also used as a check. In categories 1 and 2, it confirms that the patch does what the text says, and it can reveal a clone refactoring that the text does not even call dedup (e.g. "consolidate into the core"). In category 3, it confirms that no patch in the thread does clone refactoring or preventive reuse. Without a diff, as in a PULL, what the text states is what counts.

**1. Clone refactoring** (`clone_refactoring`)
The patch consolidates repeated code into a single place: a helper, a macro, a table, a kernel API or common code. Clone size is not considered, so small repetition also counts, such as a local variable introduced to avoid repeating the same expression, or two constants for the same thing unified into one.
- Example: `CADnq5_MGRpNE8Ge_5=w5mQDJivdS2Wt_2LD=dR-mrWH89Yc-0A@mail.gmail.com` (`[PATCH] drm/amdgpu: deduplicate ring preempt ib function`, `sample_review_all_v2_consolidated_quotes_removed.csv`, row 276)
> "The ring preemption function is identical for both gfx_v11_0 and gfx_v12_0. This patch refactors the code by moving the core logic into a generic function inside amdgpu_gfx.c to reduce code duplication"

**2. Preventive reuse** (`preventive_reuse`)
A patch avoids writing a copy, in one of these ways:
- code is moved to a common place with the stated purpose of being used by another component, even if no copy exists yet;
- the patch deliberately extends or reuses existing code instead of writing a duplicate of it, and says so (e.g. adding a mode to an existing framework instead of reimplementing what it already does elsewhere).

The diff must show that the reuse was actually done (or, without a diff, the text must say the patch does it). A reviewer contesting the choice does not change the category. If the reuse is only suggested in the discussion and no patch in the thread implements it, the thread is category 3, not 2.

The other user does not need to be in another file, driver or subsystem: code written or reorganized once so that two callers in the same driver can share it also counts (e.g. a single helper for the source and destination sides). What separates this category from category 1 is that no repeated code existed before the patch: in category 1 a copy is removed, here the copy is never written.

"Before the patch" means the kernel tree the patch applies to, not earlier versions of the same series. A copy that only existed in a previous version (v1, v2, ...) and is merged away in the current one, as in a changelog like "merged the logic of foo_std() into foo() to remove duplicate code", never reached the tree, so it is category 2, not 1.
- Example: `CAD=FV=XsUbyB4enDobda3eDoTpCqdgVogyC3YWGe9rsjgR1REw@mail.gmail.com` (`[PATCH v3 0/5] Add and enable GPI DMA users`, `sample_review_all_v2_consolidated_quotes_removed.csv`, row 168)
> "GENI_IF_DISABLE_RO is used by geni spi driver as well to check the status if GENI, so move this to common header qcom-geni-se.h"
- Example of a copy removed between versions of the series: `1663836294-5698-2-git-send-email-manikanta.guntupalli@xilinx.com` (`[PATCH V2 0/9] Added Standard mode and SMBus support.`, `sample_review_all_v2_consolidated_quotes_removed.csv`, row 29). The v1 had a separate `xiic_std_fill_tx_fifo`; in the v2 diff the existing `xiic_fill_tx_fifo` gains an `if (i2c->dynamic)` branch and serves both modes.
> "Changes for v2: Merged the logic of xiic_std_fill_tx_fifo into xiic_fill_tx_fifo to remove duplicate code."
- Example of extending an existing framework instead of duplicating it: `20151009182228.14752.99700.stgit@gimli.home` (`[RFC PATCH 0/2] VFIO no-iommu`, `sample_review_all_v2_consolidated_quotes_removed.csv`, row 52). Instead of UIO gaining its own MSI/X support, VFIO, which already has it, gains a no-iommu mode. Nothing is removed and nothing is moved to a common place.
> "VFIO on the other hand [...] provides a much more complete device interface, which already supports full MSI/X support. [...] we can at least think about doing it in a way that properly taints the kernel and avoids creating new code duplicating existing code"
- Example of reuse contested by a reviewer: `FFF73D592F13FD46B8700F0A279B802F573BC731@ORSMSX114.amr.corp.intel.com` (`[PATCH V3 0/3] iommu: Add support to change default domain of an iommu`, `sample_review_all_v2_consolidated_quotes_removed.csv`, row 189). The diff passes `dev` and `prev_dom`, already computed by the caller, into the new helper instead of looking them up again; the reviewer asks twice to drop the parameters, but the diff implements the reuse.
> "I passed it as a parameter because it's already done by iommu_group_store_type() (as below) and I thought that I could save from duplicating code by passing it as a parameter."

**3. Discussion about duplication** (`duplication_discussion`)
No patch in the thread does clone refactoring or preventive reuse, but someone discusses duplication: asks for dedup, suggests reusing something that already exists, questions, defends or accepts a copy, or attributes a bug to copies that diverged (one was fixed and the other was not). A patch that introduces duplication belongs here when someone comments on the copy, and that includes the author stating the copy (e.g. "this driver is based on foo.c", "copied from the v11 implementation", "same as X but for Y").
- Example: `20110324164137.GA22838@kroah.com` (`[PATCH] i2c/busses: Driver for Devantech USB-ISS I2C adapter`, `sample_review_all_v2_consolidated_quotes_removed.csv`, row 38). The patch adds a new I2C driver, and the USB maintainer questions it because the device already works through an existing interface. No patch in the thread does anything about it.
> "if this just duplicates the cdc-acm interface, and it looks like it does, I don't think you need a kernel driver at all for it."

**4. Self-admitted technical debt (SATD)** (`satd`)
This category can occur alone or together with any of the others (1, 2 or 3).

A code duplication is admitted as technical debt in at least one of these ways:
- **in the code:** a comment anywhere in the diff admits the duplication. E.g. `/* FIXME: duplicated from foo.c, should be shared */`. The comment does not need a TODO/FIXME/XXX/HACK tag, and it does not need to be added by the thread: an unchanged context line or a removed line also counts;
- **in the discussion:** someone in the thread admits that a duplication is debt. E.g. "copying for now, will dedup later", or "ok, I'll unify this in a follow-up".

It still counts when the same thread pays the debt (then it goes together with category 1).

It also counts when the patch itself introduces the copy and its author admits it is a stopgap, for example by calling it a "hack" or saying such code is not wanted in the proper place. A copy that is only stated, as in "copied from foo.c" with no admission that it is a problem, is category 3, not 4.

The text must name the duplication or the need to share or unify the code. It may do so implicitly, as long as the rest of the thread makes clear that the comment is about a duplication (e.g. "need to flatten these together" about two sets of definitions that the author calls "duplicated"). A TODO that only asks to move code ("move this into a common header") does not count.
- Example: `1310646827-25690-8-git-send-email-jic23@cam.ac.uk` (`[PATCH 0/7] IIO: Fix to error path and some housekeeping.`, `sample_review_all_v2_consolidated_quotes_removed.csv`, row 5), labelled `clone_refactoring,satd`. A context line in the diff of `iio.h` keeps the comment below above `enum iio_chan_type`, and patch 7/7 removes the parallel `IIO_EV_CLASS_*`/`IIO_EV_MOD_*` definitions, explaining "The original definitions were duplicated to reduce tree churn [...] Now there is no point in maintaining the two sets of definitions."
> "naughty temporary hack to match these against the event version - need to flattern these together"
- Example of a copy introduced and admitted by its author: `CAPY8ntDuKjD08Q0Y8uukpd7ep85y2qoGDv8hPFxu3QPmL8+wew@mail.gmail.com` (`[PATCH 00/18] BCM2835 DMA mapping cleanups and fixes`, `sample_review_all_v2_consolidated_quotes_removed.csv`, row 182), labelled `clone_refactoring,satd`. One patch copies the `dma-ranges` handling functions into `bcm2835-dma.c` to read `ranges` instead, and the cover letter says that code is not wanted in the core.
> "There appears to be no easy route to access "ranges", so duplicate the functions for handling "dma-ranges" here to keep the hack contained."

## What is `no`

A `no` thread has the single category **5. Not related** (`not_duplication`): it does not discuss code duplication.

- **Redundancy:** removing an unnecessary check, call, assignment or variable. Ask: if both copies stayed, could someone change one and forget the other? If the answer is "the second one simply doesn't need to exist", it is redundancy.
- **Duplication that is not code:** a duplicate table entry, a declaration repeated by mistake, a repeated word in a comment, and other senses of "duplicate", "redundant" and "repeated", such as the I2C "repeated start".
- **Copies only in the binary:** a function defined once in the source but compiled into several objects (e.g. a `static` function in a header included by several files). The source exists once, so no one can change one copy and forget the other. A thread whose only duplication remark is about this is `no` (e.g. `16aaae04-4fe8-4227-9374-0919960a4ca2@quicinc.com`, row 30 of `sample_review_all_v2_consolidated_quotes_removed.csv`: "Why are you defining functions in header causing multiple copies of them?". That thread is `yes` for other reasons, but this remark does not count toward it).
- **Relocation without a reuse motive**, including a TODO like "move to common header" that doesn't say why.
- **A PULL where only a commit title mentions dedup.** A PULL is `yes` only if the maintainer's prose states the reuse or dedup purpose.
- **A clone introduced without anyone commenting on it**: not the author, not a reviewer, and no code comment admitting the copy.
- **Dead code left behind by a refactor and removed as a fix**, with no one mentioning duplication (e.g. `20190819143515.21653-3-Bhawanpreet.Lakha@amd.com`: "during a refactor a redundant code that has unknown behaviour was added").

## General rule

Label only from the content in the sample. Evidence that exists only outside it does not count.
