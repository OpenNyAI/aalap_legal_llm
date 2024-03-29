import openai
import time
def call_openai_api(messages, max_tokens=1024, model='gpt-3.5-turbo'):
    retry_cnt = 0
    retry_limit = 3
    response = None
    while retry_cnt < retry_limit:
        try:
            completions = openai.ChatCompletion.create(model=model,
                                                       messages=messages,
                                                       max_tokens=max_tokens,
                                                       n=1,
                                                       stop=None,
                                                       temperature=0)
            response = completions.choices[0].message.content
            break
        except Exception:
            retry_cnt += 1
            time.sleep(1)
    return response

task_mapping = {'corporate_lobbying': 'ISSUE_TASK', 'learned_hands_benefits': 'ISSUE_TASK',
                'learned_hands_business': 'ISSUE_TASK', 'learned_hands_consumer': 'ISSUE_TASK',
                'learned_hands_courts': 'ISSUE_TASK', 'learned_hands_crime': 'ISSUE_TASK',
                'learned_hands_divorce': 'ISSUE_TASK', 'learned_hands_domestic_violence': 'ISSUE_TASK',
                'learned_hands_education': 'ISSUE_TASK', 'learned_hands_employment': 'ISSUE_TASK',
                'learned_hands_estates': 'ISSUE_TASK', 'learned_hands_family': 'ISSUE_TASK',
                'learned_hands_health': 'ISSUE_TASK', 'learned_hands_housing': 'ISSUE_TASK',
                'learned_hands_immigration': 'ISSUE_TASK', 'learned_hands_torts': 'ISSUE_TASK',
                'learned_hands_traffic': 'ISSUE_TASK', 'rule_qa': 'RULE_TASK',
                'international_citizenship_questions': 'RULE_TASK', 'nys_judicial_ethics': 'RULE_TASK',
                'citation_prediction_classification': 'RULE_TASK', 'citation_prediction_open': 'RULE_TASK',
                'abercrombie': 'CONCLUSION_TASK', 'diversity_1': 'CONCLUSION_TASK', 'diversity_2': 'CONCLUSION_TASK',
                'diversity_3': 'CONCLUSION_TASK', 'diversity_4': 'CONCLUSION_TASK', 'diversity_5': 'CONCLUSION_TASK',
                'diversity_6': 'CONCLUSION_TASK', 'hearsay': 'CONCLUSION_TASK',
                'personal_jurisdiction': 'CONCLUSION_TASK', 'successor_liability': 'CONCLUSION_TASK',
                'telemarketing_sales_rule': 'CONCLUSION_TASK', 'ucc_v_common_law': 'CONCLUSION_TASK',
                'consumer_contracts_qa': 'INTERPRETATION_TASK',
                'contract_nli_confidentiality_of_agreement': 'INTERPRETATION_TASK',
                'contract_nli_explicit_identification': 'INTERPRETATION_TASK',
                'contract_nli_inclusion_of_verbally_conveyed_information': 'INTERPRETATION_TASK',
                'contract_nli_limited_use': 'INTERPRETATION_TASK', 'contract_nli_no_licensing': 'INTERPRETATION_TASK',
                'contract_nli_notice_on_compelled_disclosure': 'INTERPRETATION_TASK',
                'contract_nli_permissible_acquirement_of_similar_information': 'INTERPRETATION_TASK',
                'contract_nli_permissible_copy': 'INTERPRETATION_TASK',
                'contract_nli_permissible_development_of_similar_information': 'INTERPRETATION_TASK',
                'contract_nli_permissible_post-agreement_possession': 'INTERPRETATION_TASK',
                'contract_nli_return_of_confidential_information': 'INTERPRETATION_TASK',
                'contract_nli_sharing_with_employees': 'INTERPRETATION_TASK',
                'contract_nli_sharing_with_third-parties': 'INTERPRETATION_TASK',
                'contract_nli_survival_of_obligations': 'INTERPRETATION_TASK', 'contract_qa': 'INTERPRETATION_TASK',
                'cuad_affiliate_license-licensee': 'INTERPRETATION_TASK',
                'cuad_affiliate_license-licensor': 'INTERPRETATION_TASK', 'cuad_anti-assignment': 'INTERPRETATION_TASK',
                'cuad_audit_rights': 'INTERPRETATION_TASK', 'cuad_cap_on_liability': 'INTERPRETATION_TASK',
                'cuad_change_of_control': 'INTERPRETATION_TASK',
                'cuad_competitive_restriction_exception': 'INTERPRETATION_TASK',
                'cuad_covenant_not_to_sue': 'INTERPRETATION_TASK', 'cuad_effective_date': 'INTERPRETATION_TASK',
                'cuad_exclusivity': 'INTERPRETATION_TASK', 'cuad_expiration_date': 'INTERPRETATION_TASK',
                'cuad_governing_law': 'INTERPRETATION_TASK', 'cuad_insurance': 'INTERPRETATION_TASK',
                'cuad_ip_ownership_assignment': 'INTERPRETATION_TASK',
                'cuad_irrevocable_or_perpetual_license': 'INTERPRETATION_TASK',
                'cuad_joint_ip_ownership': 'INTERPRETATION_TASK', 'cuad_license_grant': 'INTERPRETATION_TASK',
                'cuad_liquidated_damages': 'INTERPRETATION_TASK', 'cuad_minimum_commitment': 'INTERPRETATION_TASK',
                'cuad_most_favored_nation': 'INTERPRETATION_TASK',
                'cuad_no-solicit_of_customers': 'INTERPRETATION_TASK',
                'cuad_no-solicit_of_employees': 'INTERPRETATION_TASK', 'cuad_non-compete': 'INTERPRETATION_TASK',
                'cuad_non-disparagement': 'INTERPRETATION_TASK', 'cuad_non-transferable_license': 'INTERPRETATION_TASK',
                'cuad_notice_period_to_terminate_renewal': 'INTERPRETATION_TASK',
                'cuad_post-termination_services': 'INTERPRETATION_TASK',
                'cuad_price_restrictions': 'INTERPRETATION_TASK', 'cuad_renewal_term': 'INTERPRETATION_TASK',
                'cuad_revenue-profit_sharing': 'INTERPRETATION_TASK', 'cuad_rofr-rofo-rofn': 'INTERPRETATION_TASK',
                'cuad_source_code_escrow': 'INTERPRETATION_TASK',
                'cuad_termination_for_convenience': 'INTERPRETATION_TASK',
                'cuad_third_party_beneficiary': 'INTERPRETATION_TASK', 'cuad_uncapped_liability': 'INTERPRETATION_TASK',
                'cuad_unlimited-all-you-can-eat-license': 'INTERPRETATION_TASK',
                'cuad_volume_restriction': 'INTERPRETATION_TASK', 'cuad_warranty_duration': 'INTERPRETATION_TASK',
                'insurance_policy_interpretation': 'INTERPRETATION_TASK', 'jcrew_blocker': 'INTERPRETATION_TASK',
                'maud_ability_to_consummate_concept_is_subject_to_mae_carveouts': 'INTERPRETATION_TASK',
                'maud_financial_point_of_view_is_the_sole_consideration': 'INTERPRETATION_TASK',
                'maud_accuracy_of_fundamental_target_rws_bringdown_standard': 'INTERPRETATION_TASK',
                'maud_accuracy_of_target_general_rw_bringdown_timing_answer': 'INTERPRETATION_TASK',
                'maud_accuracy_of_target_capitalization_rw_(outstanding_shares)_bringdown_standard_answer': 'INTERPRETATION_TASK',
                'maud_additional_matching_rights_period_for_modifications_(cor)': 'INTERPRETATION_TASK',
                'maud_application_of_buyer_consent_requirement_(negative_interim_covenant)': 'INTERPRETATION_TASK',
                'maud_buyer_consent_requirement_(ordinary_course)': 'INTERPRETATION_TASK',
                'maud_change_in_law__subject_to_disproportionate_impact_modifier': 'INTERPRETATION_TASK',
                'maud_changes_in_gaap_or_other_accounting_principles__subject_to_disproportionate_impact_modifier': 'INTERPRETATION_TASK',
                'maud_cor_permitted_in_response_to_intervening_event': 'INTERPRETATION_TASK',
                'maud_cor_permitted_with_board_fiduciary_determination_only': 'INTERPRETATION_TASK',
                'maud_cor_standard_(intervening_event)': 'INTERPRETATION_TASK',
                'maud_cor_standard_(superior_offer)': 'INTERPRETATION_TASK',
                'maud_definition_contains_knowledge_requirement_-_answer': 'INTERPRETATION_TASK',
                'maud_definition_includes_asset_deals': 'INTERPRETATION_TASK',
                'maud_definition_includes_stock_deals': 'INTERPRETATION_TASK',
                'maud_fiduciary_exception__board_determination_standard': 'INTERPRETATION_TASK',
                'maud_fiduciary_exception_board_determination_trigger_(no_shop)': 'INTERPRETATION_TASK',
                'maud_fls_(mae)_standard': 'INTERPRETATION_TASK',
                'maud_general_economic_and_financial_conditions_subject_to_disproportionate_impact_modifier': 'INTERPRETATION_TASK',
                'maud_includes_consistent_with_past_practice': 'INTERPRETATION_TASK',
                'maud_initial_matching_rights_period_(cor)': 'INTERPRETATION_TASK',
                'maud_initial_matching_rights_period_(ftr)': 'INTERPRETATION_TASK',
                'maud_intervening_event_-_required_to_occur_after_signing_-_answer': 'INTERPRETATION_TASK',
                'maud_knowledge_definition': 'INTERPRETATION_TASK',
                'maud_liability_standard_for_no-shop_breach_by_target_non-do_representatives': 'INTERPRETATION_TASK',
                'maud_ordinary_course_efforts_standard': 'INTERPRETATION_TASK',
                'maud_pandemic_or_other_public_health_event__subject_to_disproportionate_impact_modifier': 'INTERPRETATION_TASK',
                'maud_pandemic_or_other_public_health_event_specific_reference_to_pandemic-related_governmental_responses_or_measures': 'INTERPRETATION_TASK',
                'maud_relational_language_(mae)_applies_to': 'INTERPRETATION_TASK',
                'maud_specific_performance': 'INTERPRETATION_TASK', 'maud_tail_period_length': 'INTERPRETATION_TASK',
                'maud_type_of_consideration': 'INTERPRETATION_TASK', 'opp115_data_retention': 'INTERPRETATION_TASK',
                'opp115_data_security': 'INTERPRETATION_TASK', 'opp115_do_not_track': 'INTERPRETATION_TASK',
                'opp115_first_party_collection_use': 'INTERPRETATION_TASK',
                'opp115_international_and_specific_audiences': 'INTERPRETATION_TASK',
                'opp115_policy_change': 'INTERPRETATION_TASK',
                'opp115_third_party_sharing_collection': 'INTERPRETATION_TASK',
                'opp115_user_access,_edit_and_deletion': 'INTERPRETATION_TASK',
                'opp115_user_choice_control': 'INTERPRETATION_TASK', 'privacy_policy_entailment': 'INTERPRETATION_TASK',
                'privacy_policy_qa': 'INTERPRETATION_TASK', 'proa': 'INTERPRETATION_TASK',
                'ssla_company_defendants': 'INTERPRETATION_TASK', 'ssla_individual_defendants': 'INTERPRETATION_TASK',
                'ssla_plaintiff': 'INTERPRETATION_TASK', 'sara_entailment': 'INTERPRETATION_TASK',
                'sara_numeric': 'INTERPRETATION_TASK',
                'supply_chain_disclosure_best_practice_accountability': 'INTERPRETATION_TASK',
                'supply_chain_disclosure_best_practice_audits': 'INTERPRETATION_TASK',
                'supply_chain_disclosure_best_practice_certification': 'INTERPRETATION_TASK',
                'supply_chain_disclosure_best_practice_training': 'INTERPRETATION_TASK',
                'supply_chain_disclosure_best_practice_verification': 'INTERPRETATION_TASK',
                'supply_chain_disclosure_disclosed_accountability': 'INTERPRETATION_TASK',
                'supply_chain_disclosure_disclosed_audits': 'INTERPRETATION_TASK',
                'supply_chain_disclosure_disclosed_certification': 'INTERPRETATION_TASK',
                'supply_chain_disclosure_disclosed_training': 'INTERPRETATION_TASK',
                'supply_chain_disclosure_disclosed_verification': 'INTERPRETATION_TASK',
                'unfair_tos': 'INTERPRETATION_TASK', 'canada_tax_court_outcomes': 'RHETORIC_TASK',
                'definition_classification': 'RHETORIC_TASK', 'definition_extraction': 'RHETORIC_TASK',
                'function_of_decision_section': 'RHETORIC_TASK', 'legal_reasoning_causality': 'RHETORIC_TASK',
                'oral_argument_question_purpose': 'RHETORIC_TASK', 'overruling': 'RHETORIC_TASK',
                'scalr': 'RHETORIC_TASK', 'textualism_tool_dictionaries': 'RHETORIC_TASK',
                'textualism_tool_plain': 'RHETORIC_TASK'}