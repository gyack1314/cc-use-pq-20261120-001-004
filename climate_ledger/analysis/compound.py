"""热旱复合暴露与共享设施联合行动分析。

本模块不修改任何单指标口径：温度与水量始终保留各自的原始暴露结果，
既不把摄氏度与升数相加，也不产生掩盖资料不确定性的综合分数。
联合口径只回答三类互斥问题：单独高温、单独缺水、同时危险（含恢复期遇缺水），
并把资料缺测、交集未知和事件连接未知显式保留在结果中。
"""
from datetime import date
from decimal import Decimal

from ..contracts import day as parse_day
from ..contracts import days, distinct_strings, identifier, integer, number, period, require, shape
from .exposure import Exposure


RECOVERY_DAYS = 2
STRATA = ("both", "heat_only", "water_only")


def compound_spec(value):
    """解析复合查询；联合人口与设施服务是随查询提交的业务资料，不入总账。"""
    shape(value, ("heat_metric", "water_metric", "start", "end", "zones"),
          ("minimum_coverage", "joint_population", "facilities"))
    start, end = period(value["start"], value["end"])
    heat_metric = identifier(value["heat_metric"], "heat_metric")
    water_metric = identifier(value["water_metric"], "water_metric")
    require(heat_metric != water_metric, "metric", "复合分析需要两个不同指标")
    zones = distinct_strings(value["zones"], "zones")
    coverage = str(number(value.get("minimum_coverage", "0.8"), "minimum_coverage", 0, 1))
    joint = [joint_population_record(row) for row in value.get("joint_population", [])]
    facilities = [facility_record(row) for row in value.get("facilities", [])]
    require(len({(r["zone"], r["day"], r["group"]) for r in joint}) == len(joint),
            "joint_population", "同区同日同群体的联合人口记录不能重复")
    require(len({f["facility"] for f in facilities}) == len(facilities),
            "facility", "设施编号不能重复")
    return {"heat_metric": heat_metric, "water_metric": water_metric, "start": start, "end": end,
            "zones": zones, "minimum_coverage": coverage,
            "joint_population": joint, "facilities": facilities}


def joint_population_record(value):
    shape(value, ("zone", "day", "group", "heat_affected", "water_affected"), optional=("intersection",))
    when = parse_day(value["day"], "day")
    heat_affected = integer(value["heat_affected"], "heat_affected")
    water_affected = integer(value["water_affected"], "water_affected")
    intersection = value.get("intersection")
    if intersection is not None:
        intersection = integer(intersection, "intersection")
    return {"zone": identifier(value["zone"], "zone"), "day": when,
            "group": identifier(value["group"], "group"),
            "heat_affected": heat_affected, "water_affected": water_affected,
            "intersection": intersection}


def facility_record(value):
    shape(value, ("facility", "zone", "daily_capacity", "services"))
    services = [service_record(row) for row in value["services"]]
    require(services, "facility", "设施至少声明一项服务")
    require(len({s["service"] for s in services}) == len(services), "facility", "设施内服务编号不能重复")
    return {"facility": identifier(value["facility"], "facility"),
            "zone": identifier(value["zone"], "zone"),
            "daily_capacity": integer(value["daily_capacity"], "daily_capacity", 1),
            "services": services}


def service_record(value):
    shape(value, ("service", "hazards", "cost_per_person", "groups", "priority"))
    hazards = value["hazards"]
    require(isinstance(hazards, list) and hazards and set(hazards).issubset({"heat", "water"}),
            "hazards", "服务适用危险只能是heat、water或两者")
    require(len(hazards) == len(set(hazards)), "hazards", "服务危险声明不能重复")
    groups = distinct_strings(value["groups"], "groups")
    return {"service": identifier(value["service"], "service"),
            "hazards": sorted(hazards),
            "cost_per_person": integer(value["cost_per_person"], "cost_per_person", 1),
            "groups": groups, "priority": integer(value["priority"], "priority", 0, 100)}


class Compound:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.exposure = Exposure(snapshot)

    def analyze(self, value):
        spec = compound_spec(value)
        self.exposure.readings.catalog.metric(spec["heat_metric"])
        self.exposure.readings.catalog.metric(spec["water_metric"])
        require(set(spec["zones"]).issubset(self.snapshot["zone"]), "zone", "查询含未知行政区")
        known_groups = {zone: set(self.exposure.residents.registered_groups(zone)) for zone in spec["zones"]}
        for record in spec["joint_population"]:
            require(record["zone"] in known_groups and record["group"] in known_groups[record["zone"]],
                    "joint_population", "联合人口引用了未登记的行政区或群体",
                    zone=record["zone"], group=record["group"])
        for facility in spec["facilities"]:
            require(facility["zone"] in known_groups, "facility", "设施所在行政区未登记",
                    facility=facility["facility"])
            for service in facility["services"]:
                require(set(service["groups"]).issubset(known_groups[facility["zone"]]),
                        "facility", "服务引用了未登记群体", service=service["service"])
        joint_index = {(r["zone"], r["day"], r["group"]): r for r in spec["joint_population"]}
        daily = []
        for zone in spec["zones"]:
            heat_rows, water_rows, population_rows = self.zone_inputs(zone, spec)
            timeline = self.heat_timeline(heat_rows)
            for when in days(spec["start"], spec["end"]):
                daily.append(self.zone_day(zone, when, heat_rows[when], water_rows[when],
                                           population_rows[when], timeline, joint_index))
        episodes = self.heat_episodes(daily)
        compound_events = self.compound_events(daily)
        totals = self.totals(daily)
        totals["compound_event_count"] = len(compound_events)
        qualified = totals["unknown_zone_days"] == 0 and not totals["uncertainty_reasons"]
        result = {"query": spec, "status": "qualified" if qualified else "incomplete",
                  "recovery_days": RECOVERY_DAYS,
                  "single_indicator": {
                      spec["heat_metric"]: self.exposure.analyze(self.single_query(spec, spec["heat_metric"])),
                      spec["water_metric"]: self.exposure.analyze(self.single_query(spec, spec["water_metric"]))},
                  "daily": daily, "heat_episodes": episodes, "compound_events": compound_events,
                  "joint_totals": totals,
                  "interpretation": "联合口径保留两个单指标原始结果；互斥分层不跨量纲相加，未知部分不折算为确定保护"}
        if spec["facilities"]:
            result["joint_actions"] = self.allocation_report(daily, spec)
        return result

    @staticmethod
    def single_query(spec, metric):
        return {"metric": metric, "start": spec["start"], "end": spec["end"], "zones": spec["zones"],
                "minimum_coverage": spec["minimum_coverage"]}

    def zone_inputs(self, zone, spec):
        heat_rows, water_rows, population_rows = {}, {}, {}
        for when in days(spec["start"], spec["end"]):
            heat_rows[when] = self.exposure.readings.zone_day(zone, spec["heat_metric"], when,
                                                               spec["minimum_coverage"])
            water_rows[when] = self.exposure.readings.zone_day(zone, spec["water_metric"], when,
                                                                spec["minimum_coverage"])
            population_rows[when] = self.exposure.residents.day(zone, when)
        return heat_rows, water_rows, population_rows

    @staticmethod
    def heat_timeline(heat_rows):
        """按同区连续危险日识别事件；缺测既不结束事件也不延续事件。

        状态机：
        - 危险日记 heat_event；安全日确认结束危险段，其后两个安全日记 recovery；
        - 缺测日记 heat_unknown：它不结束事件，也不证明事件继续；
        - 缺测之后、下一个确认危险日之前，恢复资格无法确认，安全日只能记 heat_safe，
          后续出现的危险段在事件层面标记为 possible_link（可能与前段相连）。
        """
        ordered = sorted(heat_rows)
        labels = {}
        last_hazard = None            # 最近一个确认危险日
        gap_lock = False             # 出现缺测后，恢复资格冻结，直到下一个确认危险日
        for when in ordered:
            row = heat_rows[when]
            if not row["usable"]:
                labels[when] = "heat_unknown"
                gap_lock = True
                continue
            if row["hazardous"]:
                labels[when] = "heat_event"
                last_hazard = when
                gap_lock = False
                continue
            # 确认安全日：明确结束当前可确认危险段
            within_recovery = (
                not gap_lock and last_hazard is not None
                and (date.fromisoformat(when) - date.fromisoformat(last_hazard)).days <= RECOVERY_DAYS)
            labels[when] = "recovery" if within_recovery else "heat_safe"
        return labels

    def zone_day(self, zone, when, heat, water, population, timeline, joint_index):
        heat_label = timeline[when]
        water_hazard = water["hazardous"] if water["usable"] else None
        reasons = []
        if not heat["usable"]:
            reasons.append("heat_unknown")
        if not water["usable"]:
            reasons.append("water_unknown")
        if not population["complete"]:
            reasons.append("population_gap")
        compound = self.compound_qualification(heat_label, water_hazard)
        groups = []
        for group in population["groups"]:
            groups.append(self.group_strata(zone, when, group, heat, water, joint_index, reasons))
        strata = self.sum_strata(groups, heat["usable"], water["usable"])
        return {"zone": zone, "day": when, "heat_phase": heat_label,
                "heat": {"usable": heat["usable"], "hazardous": heat["hazardous"],
                         "value": heat["value"], "coverage": heat["coverage"],
                         "evidence": heat["evidence"], "issues": heat["issues"],
                         "metric_revision": heat["metric_revision"]},
                "water": {"usable": water["usable"], "hazardous": water["hazardous"],
                          "value": water["value"], "coverage": water["coverage"],
                          "evidence": water["evidence"], "issues": water["issues"],
                          "metric_revision": water["metric_revision"]},
                "population": population, "compound_qualified": compound["qualified"],
                "compound_status": compound["status"], "compound_reason": compound["reason"],
                "strata": strata, "groups": groups, "unknown_reasons": sorted(set(reasons))}

    @staticmethod
    def compound_qualification(heat_label, water_hazard):
        """缺水日只有落在热事件或恢复期内才属于复合暴露。"""
        if water_hazard is None:
            return {"qualified": None, "status": "unknown", "reason": "water_unknown"}
        if not water_hazard:
            return {"qualified": False, "status": "not_water_hazard", "reason": "water_safe"}
        if heat_label in ("heat_event", "recovery"):
            return {"qualified": True, "status": "confirmed",
                    "reason": "compound" if heat_label == "heat_event" else "recovery_water"}
        if heat_label == "heat_unknown":
            return {"qualified": None, "status": "possible", "reason": "heat_connection_unknown"}
        return {"qualified": False, "status": "not_compound", "reason": "outside_heat_window"}

    @staticmethod
    def group_strata(zone, when, group, heat, water, joint_index, reasons):
        record = joint_index.get((zone, when, group["group"]))
        base = {"group": group["group"], "population": group["count"],
                "vulnerability": group["vulnerability"], "joint_input": None,
                "intersection_known": False, "strata": None, "uncertainty": []}
        if not heat["usable"] or not water["usable"]:
            base["uncertainty"].append("hazard_status_unknown")
            return base
        if record is None:
            base["uncertainty"].append("joint_population_gap")
            reasons.append("joint_population_gap")
            return base
        Compound.check_hazard_consistency(zone, when, group["group"], record, heat, water)
        base["joint_input"] = {"heat_affected": record["heat_affected"],
                               "water_affected": record["water_affected"],
                               "intersection": record["intersection"]}
        count = group["count"]
        heat_affected, water_affected = record["heat_affected"], record["water_affected"]
        if not (0 <= heat_affected <= count and 0 <= water_affected <= count):
            require(False, "joint_population", "联合人群数超过该群体人口分母",
                    zone=zone, day=when, group=group["group"])
        if record["intersection"] is not None:
            both = record["intersection"]
            if not (max(0, heat_affected + water_affected - count) <= both
                    <= min(heat_affected, water_affected)):
                require(False, "joint_population", "交集人数与两个总数及人口分母不相容",
                        zone=zone, day=when, group=group["group"])
            base["intersection_known"] = True
            base["strata"] = Compound.fixed_strata(heat_affected, water_affected, both)
            return base
        lower = max(0, heat_affected + water_affected - count)
        upper = min(heat_affected, water_affected)
        base["uncertainty"].append("intersection_unknown")
        base["strata"] = {
            "both": {"lower": lower, "upper": upper},
            "heat_only": {"lower": heat_affected - upper, "upper": heat_affected - lower},
            "water_only": {"lower": water_affected - upper, "upper": water_affected - lower},
            "neither": {"lower": count - heat_affected - water_affected + lower,
                        "upper": count - heat_affected - water_affected + upper}}
        return base

    @staticmethod
    def check_hazard_consistency(zone, when, group_name, record, heat, water):
        """业务联合人数必须与观测到的危险标志相容：安全日不能有受影响人数。"""
        if not heat["hazardous"] and record["heat_affected"]:
            require(False, "joint_population", "热观测未达危险阈值，不能登记受热人数",
                    zone=zone, day=when, group=group_name)
        if not water["hazardous"] and record["water_affected"]:
            require(False, "joint_population", "缺水观测未达危险阈值，不能登记缺水人数",
                    zone=zone, day=when, group=group_name)

    @staticmethod
    def fixed_strata(heat_affected, water_affected, both):
        return {"both": both, "heat_only": heat_affected - both,
                "water_only": water_affected - both,
                "neither": None}

    @staticmethod
    def sum_strata(groups, heat_usable, water_usable):
        """同一居民同一天只能进入一次联合分母：三类互斥分层相加可核对。

        只要有一个群体交集未知，整日结果就只给可行上下界；已知群体按确定值
        同时计入上下界，避免把未知群体折算成确定人数。
        """
        if not heat_usable or not water_usable:
            return None
        if any(group["strata"] is None for group in groups):
            return None

        def value_of(group, stratum, bound):
            item = group["strata"][stratum]
            return item if isinstance(item, int) else item[bound]

        if any(not group["intersection_known"] for group in groups):
            return {stratum: {"lower": sum(value_of(group, stratum, "lower") for group in groups),
                              "upper": sum(value_of(group, stratum, "upper") for group in groups)}
                    for stratum in STRATA}
        return {stratum: sum(group["strata"][stratum] for group in groups) for stratum in STRATA}

    @staticmethod
    def heat_episodes(daily):
        """每个连续危险日段是一个热事件，确认结束后的恢复期附属于该事件。

        缺测把危险段切开：前段右边界标 unknown，后段左边界标 possible_link，
        表示两段可能相连，但既不宣称已连接也不宣称已断开。恢复期在第二个恢复日
        之后或下一个安全日自动收尾。
        """
        result = []
        for zone in sorted({row["zone"] for row in daily}):
            rows = sorted((r for r in daily if r["zone"] == zone), key=lambda r: r["day"])
            current = None
            possible_link = False
            for row in rows:
                phase = row["heat_phase"]
                if phase == "heat_event":
                    if current is None:
                        current = {"zone": zone, "start": row["day"], "end": row["day"],
                                   "hazard_days": 0, "recovery_days": [],
                                   "left_boundary": "possible_link" if possible_link else "window",
                                   "right_boundary": "window", "connection": "possible" if possible_link else "confirmed"}
                    current["end"] = row["day"]
                    current["hazard_days"] += 1
                elif phase == "heat_unknown":
                    if current is not None:
                        current["right_boundary"] = "unknown"
                        result.append(current)
                        current = None
                        possible_link = True
                else:
                    if current is not None:
                        if phase == "recovery":
                            current["recovery_days"].append(row["day"])
                            current["end"] = row["day"]
                            if len(current["recovery_days"]) == RECOVERY_DAYS:
                                current["right_boundary"] = "observed"
                                result.append(current)
                                current = None
                                possible_link = False
                        else:
                            current["right_boundary"] = "observed"
                            result.append(current)
                            current = None
                            possible_link = False
            if current is not None:
                result.append(current)
        return result

    @staticmethod
    def compound_events(daily):
        """把连续的复合/可能复合缺水日归为同一复合事件，并保留依据。"""
        result = []
        for zone in sorted({row["zone"] for row in daily}):
            rows = sorted((r for r in daily if r["zone"] == zone), key=lambda r: r["day"])
            current = None
            for row in rows:
                active = row["compound_qualified"] is True or (
                    row["compound_qualified"] is None and row["water"]["hazardous"] is True)
                if not active:
                    if current is not None:
                        result.append(current)
                        current = None
                    continue
                if current is None:
                    current = {"zone": zone, "start": row["day"], "end": row["day"], "days": 0,
                               "confirmed_days": 0, "possible_days": 0, "heat_phases": [],
                               "boundary_unknown": False, "days_detail": []}
                current["end"] = row["day"]
                current["days"] += 1
                current["heat_phases"].append(row["heat_phase"])
                if row["compound_qualified"] is True:
                    current["confirmed_days"] += 1
                else:
                    current["possible_days"] += 1
                    current["boundary_unknown"] = True
                current["days_detail"].append({
                    "day": row["day"], "heat_phase": row["heat_phase"],
                    "compound_status": row["compound_status"],
                    "water_value": row["water"]["value"], "heat_value": row["heat"]["value"]})
            if current is not None:
                result.append(current)
        return result

    @staticmethod
    def totals(daily):
        """互斥分层汇总：同一人同一天只进入一次分母，各分层可相互核对。

        交集已知的区日计入 confirmed 口径；交集未知的区日只给可行上下界，
        计入 bounded 口径，绝不折算为单一确定保护数。
        """
        totals = {
            "zone_days": len(daily), "unknown_zone_days": 0,
            "confirmed": {stratum: 0 for stratum in STRATA},
            "bounded": {stratum: {"lower": 0, "upper": 0} for stratum in STRATA},
            "heat_affected_person_days": {"confirmed": 0, "bounded": {"lower": 0, "upper": 0}},
            "water_affected_person_days": {"confirmed": 0, "bounded": {"lower": 0, "upper": 0}},
            "compound_person_days": {"confirmed": 0, "bounded": {"lower": 0, "upper": 0},
                                     "recovery_water_confirmed": 0,
                                     "recovery_water_bounded": {"lower": 0, "upper": 0}},
            "uncertainty_reasons": []}
        reasons = set()
        for row in daily:
            if not row["heat"]["usable"] or not row["water"]["usable"] or not row["population"]["complete"]:
                totals["unknown_zone_days"] += 1
            reasons.update(row["unknown_reasons"])
            for group in row["groups"]:
                reasons.update(group["uncertainty"])
            if row["strata"] is None:
                continue
            recovery_water = row["compound_reason"] == "recovery_water"
            if all(g["intersection_known"] for g in row["groups"]):
                for stratum in STRATA:
                    totals["confirmed"][stratum] += row["strata"][stratum]
                totals["heat_affected_person_days"]["confirmed"] += row["strata"]["both"] + row["strata"]["heat_only"]
                totals["water_affected_person_days"]["confirmed"] += row["strata"]["both"] + row["strata"]["water_only"]
                if row["compound_qualified"] is True:
                    totals["compound_person_days"]["confirmed"] += row["strata"]["both"]
                    if recovery_water:
                        totals["compound_person_days"]["recovery_water_confirmed"] += row["strata"]["water_only"]
                        totals["compound_person_days"]["confirmed"] += row["strata"]["water_only"]
            else:
                for stratum in STRATA:
                    totals["bounded"][stratum]["lower"] += row["strata"][stratum]["lower"]
                    totals["bounded"][stratum]["upper"] += row["strata"][stratum]["upper"]
                heat_both = row["strata"]["both"]
                heat_only = row["strata"]["heat_only"]
                water_only = row["strata"]["water_only"]
                totals["heat_affected_person_days"]["bounded"]["lower"] += heat_both["lower"] + heat_only["lower"]
                totals["heat_affected_person_days"]["bounded"]["upper"] += heat_both["upper"] + heat_only["upper"]
                totals["water_affected_person_days"]["bounded"]["lower"] += heat_both["lower"] + water_only["lower"]
                totals["water_affected_person_days"]["bounded"]["upper"] += heat_both["upper"] + water_only["upper"]
                if row["compound_qualified"] is True:
                    lower = heat_both["lower"] + (water_only["lower"] if recovery_water else 0)
                    upper = heat_both["upper"] + (water_only["upper"] if recovery_water else 0)
                    totals["compound_person_days"]["bounded"]["lower"] += lower
                    totals["compound_person_days"]["bounded"]["upper"] += upper
                    if recovery_water:
                        totals["compound_person_days"]["recovery_water_bounded"]["lower"] += water_only["lower"]
                        totals["compound_person_days"]["recovery_water_bounded"]["upper"] += water_only["upper"]
        totals["uncertainty_reasons"] = sorted(reasons)
        return totals

    # ------------------------------------------------------------------
    # 共享设施联合行动：同一居民每日最多被服务一次，名额跨行动共享
    # ------------------------------------------------------------------
    def allocation_report(self, daily, spec):
        facility_ids = sorted(f["facility"] for f in spec["facilities"])
        report_days = []
        for row in daily:
            zone_facilities = [f for f in spec["facilities"] if f["zone"] == row["zone"]]
            if not zone_facilities or row["strata"] is None:
                continue
            report_days.append(self.allocate_day(row, zone_facilities))
        return {"facilities": facility_ids, "days": report_days,
                "allocation_rule": "按服务优先级与群体脆弱度分配；联合服务一人占用两份名额，"
                                   "同一居民每日只计入一次保护，设施总名额在所有服务间共享"}

    def allocate_day(self, row, facilities):
        known = all(g["intersection_known"] for g in row["groups"])
        if known:
            allocation = self.allocate_deterministic(row, facilities, mode="confirmed")
            return {"zone": row["zone"], "day": row["day"], "intersection_known": True,
                    "allocation": allocation}
        lower = self.allocate_deterministic(row, facilities, mode="lower_bound")
        upper = self.allocate_deterministic(row, facilities, mode="upper_bound")
        return {"zone": row["zone"], "day": row["day"], "intersection_known": False,
                "bounds": {"lower": lower, "upper": upper},
                "uncertainty": ["intersection_unknown"],
                "note": "交集未知时只给可行保护上下界，不把任一假设组合当作已兑现保护"}

    def allocate_deterministic(self, row, facilities, mode):
        capacities = {f["facility"]: f["daily_capacity"] for f in facilities}
        # 每人每日只能被服务一次：按群体与互斥分层记录已服务状态
        served = {group["group"]: {stratum: 0 for stratum in STRATA} for group in row["groups"]}
        service_log = []
        queue = sorted(((facility, service) for facility in facilities for service in facility["services"]),
                       key=lambda item: (-item[1]["priority"], item[1]["service"]))
        for facility, service in queue:
            remaining = capacities[facility["facility"]]
            assignments = []
            slots_used = 0
            eligible_groups = sorted((g for g in row["groups"] if g["group"] in service["groups"]),
                                     key=lambda g: (-Decimal(g["vulnerability"]), g["group"]))
            for group in eligible_groups:
                for stratum in self.eligible_strata(service):
                    need = self.stratum_count(group, stratum, mode) - served[group["group"]][stratum]
                    take = min(need, remaining // service["cost_per_person"])
                    if take:
                        served[group["group"]][stratum] += take
                        remaining -= take * service["cost_per_person"]
                        slots_used += take * service["cost_per_person"]
                        assignments.append({"group": group["group"], "stratum": stratum, "count": take})
            capacities[facility["facility"]] = remaining
            service_log.append({"service": service["service"], "facility": facility["facility"],
                                "hazards": service["hazards"], "cost_per_person": service["cost_per_person"],
                                "used_slots": slots_used,
                                "remaining_slots": remaining, "assignments": assignments})
        protected = {stratum: sum(served[g["group"]][stratum] for g in row["groups"]) for stratum in STRATA}
        return {"mode": mode, "protected_by_stratum": protected,
                "protected_total": sum(protected.values()), "services": service_log,
                "facility_remaining": capacities}

    @staticmethod
    def eligible_strata(service):
        """服务分层与声明危险一致：联合服务只面向同时受两类危险的人。"""
        if set(service["hazards"]) == {"heat", "water"}:
            return ["both"]
        if service["hazards"] == ["heat"]:
            return ["both", "heat_only"]
        return ["both", "water_only"]

    @staticmethod
    def stratum_count(group, stratum, mode):
        strata = group["strata"]
        if isinstance(strata[stratum], int):
            return strata[stratum]
        return strata[stratum]["upper" if mode == "upper_bound" else "lower"]


def compound_plan_spec(value):
    """复合预算组合：候选是查询内声明的设施包，按联合实际可行性评价。"""
    shape(value, ("query", "budget_cents", "packages"))
    require(isinstance(value["packages"], list) and 0 < len(value["packages"]) <= 16,
            "packages", "一次组合比较最多支持16个候选设施包")
    budget = integer(value["budget_cents"], "budget_cents", 0, 100_000_000_000)
    packages = []
    for item in value["packages"]:
        shape(item, ("package", "cost_cents"))
        packages.append({"package": identifier(item["package"], "package"),
                         "cost_cents": integer(item["cost_cents"], "cost_cents", 1, 100_000_000_000)})
    require(len({p["package"] for p in packages}) == len(packages), "packages", "候选设施包不能重复")
    return {"query": compound_spec(value["query"]), "budget_cents": budget,
            "packages": sorted(packages, key=lambda p: p["package"])}


class CompoundPlan:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def plan(self, value):
        spec = compound_plan_spec(value)
        analysis = Compound(self.snapshot).analyze(spec["query"])
        candidates, excluded = [], []
        for package in spec["packages"]:
            if package["cost_cents"] > spec["budget_cents"]:
                excluded.append({**package, "reason": "over_budget"})
            else:
                candidates.append(package)
        evaluations = []
        for package in candidates:
            package_facilities = [f for f in spec["query"]["facilities"] if f["facility"] == package["package"]]
            if not package_facilities:
                excluded.append({**package, "reason": "facility_not_declared"})
                continue
            evaluations.append(self.evaluate(analysis, package, package_facilities))
        baseline = self.null_evaluation(analysis)
        ranking = sorted(evaluations, key=lambda item: (-item["protected_lower"], item["cost_cents"], item["package"]))
        inconclusive = self.inconclusive_pairs(evaluations)
        return {"request": spec, "status": "hypothetical",
                "baseline": baseline, "evaluations": evaluations, "ranking": [item["package"] for item in ranking],
                "selected": ranking[0]["package"] if ranking else None,
                "ranking_inconclusive": inconclusive, "excluded": excluded,
                "interpretation": "候选按联合服务的实际可行性评价；交集或事件连接未知时只比较保护下界，"
                                   "假设组合不写入总账，也不代表已兑现保护"}

    def evaluate(self, analysis, package, facilities):
        day_outcomes = []
        lower_total = upper_total = 0
        unknown_days = 0
        for row in analysis["daily"]:
            if row["strata"] is None or not any(f["zone"] == row["zone"] for f in facilities):
                continue
            result = Compound(self.snapshot).allocate_day(row, facilities)
            if result["intersection_known"]:
                lower_total += result["allocation"]["protected_total"]
                upper_total = lower_total
                day_outcomes.append({"zone": row["zone"], "day": row["day"],
                                     "protected": result["allocation"]["protected_total"]})
            else:
                unknown_days += 1
                lower_total += result["bounds"]["lower"]["protected_total"]
                upper_total += result["bounds"]["upper"]["protected_total"]
                day_outcomes.append({"zone": row["zone"], "day": row["day"],
                                     "protected_lower": result["bounds"]["lower"]["protected_total"],
                                     "protected_upper": result["bounds"]["upper"]["protected_total"]})
        return {"package": package["package"], "cost_cents": package["cost_cents"],
                "protected_lower": lower_total, "protected_upper": upper_total,
                "unknown_days": unknown_days, "days": day_outcomes}

    @staticmethod
    def null_evaluation(analysis):
        return {"protected_lower": 0, "protected_upper": 0,
                "unknown_zone_days": analysis["joint_totals"]["unknown_zone_days"]}

    @staticmethod
    def inconclusive_pairs(evaluations):
        result = []
        for index, left in enumerate(evaluations):
            for right in evaluations[index + 1:]:
                if left["protected_lower"] <= right["protected_upper"] and right["protected_lower"] <= left["protected_upper"]:
                    result.append(sorted([left["package"], right["package"]]))
        return sorted(result)
